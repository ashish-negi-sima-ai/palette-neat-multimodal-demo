// webrtc.js - one permanent Insight connection per SOURCE.
//
// Adapted from the Insight vf viewer (static/viewer-react.js in the installed
// neat-insight 0.0.7).  Every step below is the sequence that viewer performs;
// what is dropped is the parts of it this demo does not have - channel
// pagination, the settings modal, ROI drawing, the React shell.  What is kept is
// the whole media path:
//
//   1. RTCPeerConnection with a STUN server
//   2. addTransceiver("video", {direction:"recvonly"})
//   3. receiver.jitterBufferTarget = videoSyncBufferMs
//        Holds video back so the metadata for a frame has arrived by the time
//        the frame is presented.  This is the whole reason overlays line up -
//        and it is also, millisecond for millisecond, the lag between a hand
//        moving and the picture showing it.  Insight's viewer ships 350 ms;
//        this demo configures 120 (config/ui_config.json render._latency),
//        because video and metadata for the same frame leave the same host at
//        the same instant and only LAN jitter has to be covered.  The status
//        line reports the buffer the browser actually applied, next to the
//        measured fps, so "the video looks slow" can be read rather than
//        guessed.
//   4. createDataChannel("metadata", {ordered:true, maxRetransmits:0})
//   5. POST <origin>/offer?channel=N with the local description as JSON
//        Insight's viewer posts this to vf directly; here it goes to this app's
//        own origin and is proxied to vf unchanged.  Signalling only - the media
//        that follows goes straight from vf to this browser.
//   6. setRemoteDescription(answer)
//
// PAIRING (spec section 13: video and metadata must always move together)
//
// vf stamps every metadata message with `_insight.rtp_timestamp`, the RTP
// timestamp of the frame the message describes.  Messages are stored in a Map
// keyed on that value; each presented frame reports its own `rtpTimestamp`
// through requestVideoFrameCallback, and the frame takes its own message out of
// the Map.  So a message can only ever be drawn on the frame it belongs to, on
// the connection it arrived on.  There is no code path by which source 1's
// metadata could reach source 0's picture: they are different
// RTCPeerConnections with different Maps.
//
// The one addition to Insight's matcher is `metadataHoldMs`: when a frame finds
// no message of its own, the previous match may be redrawn for that long.  It
// makes a presentation steady across an occasional miss and cannot invent
// anything, because it only ever redraws a message that did match a frame.
// Set it to 0 for Insight's exact behaviour.

(function (global) {
  "use strict";

  // Insight's cap on pending metadata (Vu in viewer-react.js).
  var MAX_PENDING = 300;
  var RETRY_MS = 2000;          // vf said 503: no codec identified yet
  var RETRY_FAIL_MS = 5000;     // the connection dropped
  var STALL_MS = 1800;          // frames arriving but not decoding

  // A connection can be `connected` and carry no media at all.  vf attaches a
  // channel's forwarding to ONE peer, so anything else that offers on the same
  // channel - Insight's own viewer, a second copy of this page, a diagnostic
  // tool - takes the media over.  The displaced peer is not told: it stays
  // `connected`, no state change fires, and `framesReceived` simply stops
  // growing, which the not-decoding detector above cannot see because it needs
  // frames to be arriving.
  //
  // SILENCE IS NOT DISPLACEMENT, AND THIS IS WHERE THAT WENT WRONG
  //
  // Silence used to be enough: a connection that presented no frame for 4 s was
  // torn down and re-offered.  On this board that is a blackout generator, and
  // it was the second half of the reported one - a panel going dark a moment
  // AFTER a spoken command was applied, one panel then the other, each
  // recovering at a different time.
  //
  // The mechanism: while the voice models ran on the DevKit, a spoken command
  // stopped BOTH cameras for 2.0-2.75 s and restoring them took another
  // 0.8-2.2 s, staggered per stream - comfortably more than 4 s without a
  // presented frame.  The old code then asked the server whether the source was
  // sending, and by the time the answer came back it WAS sending again, so it
  // concluded "we were displaced" and re-offered: pc.close(), a new
  // RTCPeerConnection, a new /offer, a new srcObject.  The picture the board
  // was about to bring back by itself was thrown away and rebuilt instead.
  //
  // So re-offering now requires PROOF of displacement, not the absence of
  // frames:
  //
  //   the application declared a pause    do nothing at all.  Only the code
  //                                       that CAUSES a pause can know about
  //                                       one, so it says so (expectSourcePause).
  //   vf reports another peer on the       re-offer.  Someone really did take
  //   channel (`peers > 1`)                the channel, and offering again is
  //                                        the only thing that takes it back.
  //   we never had a frame at all          re-offer.  This is the initial
  //                                        connect; there is no picture to lose.
  //   anything else                        WAIT.  We are the channel's only
  //                                        peer, so re-offering cannot change
  //                                        who receives the media - it can only
  //                                        blank a panel that was coming back.
  var MEDIA_STALL_MS = 9000;    // had frames, then nothing.  Above the worst
                                // measured camera pause plus restore, so the
                                // declared-pause window below is a second line
                                // of defence and not the only one.
  var MEDIA_NEVER_MS = 12000;   // never had a frame; slower, so an idle
                                // pipeline is not hammered with offers

  // ---- END-TO-END LATENCY, per presented frame ---------------------------
  //
  // The vision binary stamps every metadata message with `timing.pull_wall_ms`,
  // the DevKit wall clock when that frame reached the application. vf pairs the
  // message with its own video frame, and the frame that is presented takes
  // exactly that message (rtpTimestamp match), so for the SAME frame:
  //
  //   browserMs   expectedDisplayTime - receiveTime        clock-independent
  //   receiveMs   wall(receiveTime)   - pull_wall_ms        needs synced clocks
  //   displayMs   wall(expectedDisplayTime) - pull_wall_ms  needs synced clocks
  //   boardMs     send_wall_ms - pull_wall_ms               board only
  //
  // receiveMs / displayMs assume this browser's clock agrees with the DevKit's
  // (the DevKit runs chrony). `window.demoLatency()` says so next to the numbers.
  var LATENCY_WINDOW = 150;
  var latencySamples = {};

  function latencySample(channel, frameMeta, message, rtp) {
    if (!frameMeta || !message) return;
    var insight = message._insight || {};
    if (insight.rtp_timestamp === undefined || (insight.rtp_timestamp >>> 0) !== (rtp >>> 0)) return;
    var timing = (message.data && message.data.timing) || message.timing;
    if (!timing || !(timing.pull_wall_ms > 0)) return;
    var origin = performance.timeOrigin;
    var s = { at: Date.now(), browserMs: null, receiveMs: null, displayMs: null,
              boardMs: timing.send_wall_ms - timing.pull_wall_ms };
    if (typeof frameMeta.receiveTime === "number") {
      s.receiveMs = origin + frameMeta.receiveTime - timing.pull_wall_ms;
    }
    if (typeof frameMeta.expectedDisplayTime === "number") {
      s.displayMs = origin + frameMeta.expectedDisplayTime - timing.pull_wall_ms;
      if (typeof frameMeta.receiveTime === "number") {
        s.browserMs = frameMeta.expectedDisplayTime - frameMeta.receiveTime;
      }
    }
    var list = latencySamples[channel] || (latencySamples[channel] = []);
    list.push(s);
    if (list.length > LATENCY_WINDOW) list.splice(0, list.length - LATENCY_WINDOW);
  }

  function summarize(values) {
    var v = values.filter(function (x) { return typeof x === "number" && isFinite(x); })
                  .sort(function (a, b) { return a - b; });
    if (!v.length) return null;
    var pick = function (f) { return Math.round(v[Math.min(v.length - 1, Math.round(f * (v.length - 1)))] * 10) / 10; };
    return { n: v.length, min: pick(0), median: pick(0.5), p90: pick(0.9), max: pick(1) };
  }

  global.demoLatency = function () {
    var out = {
      at: new Date().toISOString(),
      note: "browserMs (receive->expected display) is exact. receiveMs/displayMs are " +
            "DevKit capture-pull to this browser and assume this machine's clock matches " +
            "the DevKit's (chrony on the DevKit). boardMs is pull->metadata send on the " +
            "DevKit. None includes sensor exposure->pull.",
      channels: {}
    };
    Object.keys(latencySamples).forEach(function (ch) {
      var list = latencySamples[ch];
      out.channels[ch] = {
        samples: list.length,
        browserMs: summarize(list.map(function (s) { return s.browserMs; })),
        receiveMs: summarize(list.map(function (s) { return s.receiveMs; })),
        displayMs: summarize(list.map(function (s) { return s.displayMs; })),
        boardMs: summarize(list.map(function (s) { return s.boardMs; }))
      };
    });
    return out;
  };

  // Diagnostic override for A/B tests: ?syncBufferMs=N on the page URL.
  var URL_SYNC_BUFFER_MS = (function () {
    try {
      var v = new URLSearchParams(global.location.search).get("syncBufferMs");
      return v === null || v === "" || isNaN(Number(v)) ? null : Number(v);
    } catch (e) { return null; }
  })();

  function newStore() {
    return {
      timestamped: new Map(),
      arrival: [],
      last: null,
      stats: { matches: 0, arrivalFallbacks: 0, misses: 0, holds: 0,
               received: 0, expired: 0, evicted: 0 }
    };
  }

  // --- from rp() in viewer-react.js -------------------------------------
  function storeMetadata(store, message, now) {
    store.stats.received += 1;
    var rtp = message && message._insight && message._insight.rtp_timestamp;
    var entry = { receivedAt: now, data: message };
    if (Number.isInteger(rtp) && rtp >= 0) {
      var key = rtp >>> 0;
      store.timestamped.delete(key);
      store.timestamped.set(key, entry);
      while (store.timestamped.size > MAX_PENDING) {
        store.timestamped.delete(store.timestamped.keys().next().value);
        store.stats.evicted += 1;
      }
      return;
    }
    store.arrival.push(entry);
    if (store.arrival.length > MAX_PENDING) {
      var over = store.arrival.length - MAX_PENDING;
      store.arrival.splice(0, over);
      store.stats.evicted += over;
    }
  }

  // --- from up() in viewer-react.js -------------------------------------
  function expire(store, retentionMs, now) {
    if (!(retentionMs > 0)) return;
    for (var pair of store.timestamped) {
      if (now - pair[1].receivedAt <= retentionMs) break;
      store.timestamped.delete(pair[0]);
      store.stats.expired += 1;
    }
    while (store.arrival.length && now - store.arrival[0].receivedAt > retentionMs) {
      store.arrival.shift();
      store.stats.expired += 1;
    }
  }

  // --- from lp() in viewer-react.js, plus the hold ----------------------
  function takeMetadata(store, rtpTimestamp, retentionMs, now, holdMs) {
    expire(store, retentionMs, now);
    var timestamped = Number.isInteger(rtpTimestamp) && rtpTimestamp >= 0;

    if (timestamped) {
      var key = rtpTimestamp >>> 0;
      var hit = store.timestamped.get(key) || null;
      if (hit) {
        store.timestamped.delete(key);
        store.stats.matches += 1;
        store.last = { data: hit.data, at: now };
        return hit;
      }
    } else {
      // No frame timestamp: newest wins, exactly as Insight does.
      var newest = store.arrival.length ? store.arrival[store.arrival.length - 1] : null;
      store.timestamped.forEach(function (entry) {
        if (!newest || entry.receivedAt >= newest.receivedAt) newest = entry;
      });
      store.timestamped.clear();
      store.arrival.length = 0;
      if (newest) {
        store.stats.arrivalFallbacks += 1;
        store.last = { data: newest.data, at: now };
        return newest;
      }
    }

    if (store.arrival.length > 0) {
      var fallback = store.arrival[store.arrival.length - 1];
      store.arrival.length = 0;
      store.stats.arrivalFallbacks += 1;
      store.last = { data: fallback.data, at: now };
      return fallback;
    }

    // The hold: redraw the last matched message for a short while so one
    // missed frame does not blink the overlay off during a presentation.
    if (holdMs > 0 && store.last && now - store.last.at <= holdMs) {
      store.stats.holds += 1;
      return { receivedAt: store.last.at, data: store.last.data };
    }

    store.stats.misses += 1;
    return null;
  }

  // --- from vp() in viewer-react.js -------------------------------------
  function negotiate(pc, url) {
    return pc.createOffer()
      .then(function (offer) { return pc.setLocalDescription(offer).then(function () { return offer; }); })
      .then(function (offer) {
        return fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(offer)
        });
      })
      .then(function (response) {
        if (!response.ok) {
          var error = new Error("HTTP " + response.status);
          error.status = response.status;
          return response.text().then(function (text) {
            error.detail = text;
            throw error;
          }, function () { throw error; });
        }
        return response.json();
      });
  }

  function retryable(status) {
    // viewer-react.js hp(): no status (network) or 5xx is worth retrying.
    // 415 is the browser having no decoder for the codec and is permanent.
    return status === undefined || status >= 500;
  }

  /**
   * Open one source's connection and keep it open.
   *
   * opts.channel            vf channel index (0 or 1 here)
   * opts.video              the <video> element for this SOURCE, forever
   * opts.onFrame(payload)   called per presented frame with the matched
   *                         metadata message (or null) - the caller draws
   * opts.onStatus(status)   {phase, text, codec, width, height, fps, kbps,
   *                          bufferedMs, targetBufferMs, framesDropped,
   *                          presentedFps, drawCallbackFps, mediaRate}
   *                          `fps` is the DECODER rate from getStats;
   *                          `presentedFps` is what reaches the screen.
   * opts.syncBufferMs       receiver.jitterBufferTarget
   * opts.retentionMs        metadata expiry, 0 = no expiry (Insight default)
   * opts.holdMs             redraw window for the last matched message
   */
  function openSource(opts) {
    if (URL_SYNC_BUFFER_MS !== null) opts.syncBufferMs = URL_SYNC_BUFFER_MS;
    var channel = opts.channel;
    var video = opts.video;
    var store = newStore();
    var live = true;
    var generation = 0;
    var pc = null;
    var dc = null;
    var receiver = null;
    var statsTimer = null;
    var retryTimer = null;
    var frameHandle = null;
    var rafHandle = null;
    var decoderHealth = { framesReceived: null, framesDecoded: null, stalledSince: null };
    // STAGE 6, THE ONE THE DECODER STATISTIC CANNOT SEE.
    //
    // `framesPerSecond` in getStats is frames the DECODER finished.  It counts
    // a re-encoded identical frame exactly like a new one, and it says nothing
    // about how often the picture reaches the screen.  Every stage upstream of
    // the browser measures ~19-20 fps of genuinely new pictures (the RTP
    // capture in scripts/fps_forensics.py and the real WebRTC peer in
    // scripts/webrtc_probe.py both show zero duplicate pictures), so if the
    // picture looks slower than that, the gap is HERE - between decode and
    // composition.
    //
    // requestVideoFrameCallback's metadata answers it directly:
    //   presentedFrames  the browser's own count of frames submitted for
    //                    composition.  Its rate IS the visible frame rate.
    //   mediaTime        the frame's position in the stream's own timeline.
    //                    Advancing slower than wall clock means playback is
    //                    running slow, not that frames are missing.
    // Neither is our own bookkeeping, and neither is derived from the other.
    var presentation = {
        callbacks: 0,           // how often WE were handed a frame to draw
        presentedFrames: null,  // the browser's counter, last seen
        mediaTime: null,
        sampledAt: null,
        lastPresentedSample: null,
        lastCallbackSample: 0,
        lastMediaSample: null,
        presentedFps: null,
        callbackFps: null,
        mediaRate: null,
        browserMs: null,        // receive -> expected display, per frame
        browserMsMax: null,
        decodeMs: null
    };

    // For the presentation delay the receiver really applied: getStats gives
    // jitterBufferDelay as a running total in seconds against a count of
    // frames, so the average is the difference of both between two samples -
    // never the totals, which would average over the whole session and hide a
    // change.
    var jitter = { delay: null, count: null };
    var meter = { bytes: null, ts: null, messages: 0, codec: null };
    // `pauseUntil` is the declared-pause window: the application telling this
    // source "the camera is about to stop on purpose, do not interpret the
    // silence".  Nothing measures it - it is asserted by the code that CAUSED
    // the pause, which is the only place that can know.
    var media = { lastFrameAt: 0, connectedAt: 0, hadFrames: false,
                  pauseUntil: 0, pauseReason: null, declaredPauses: 0 };
    var status = { phase: "connecting", text: "connecting", codec: null,
                   width: null, height: null, fps: null, kbps: null,
                   bufferedMs: null, targetBufferMs: opts.syncBufferMs,
                   framesDropped: null, presentedFps: null,
                   drawCallbackFps: null, mediaRate: null,
                   browserLatencyMs: null, browserLatencyMaxMs: null,
                   decodeMs: null,
                   reconnects: 0, mediaRecoveries: 0, sourcePauses: 0,
                   declaredPauses: 0, silenceWaits: 0 };

    // ---- diagnostics ---------------------------------------------------
    // Every media-lifecycle fact this source can observe, in one place, so
    // "did a filter command disturb the video?" is answered by counters
    // instead of by reading code.  peersCreated and offersSent are the two
    // that matter: a command that changes only rendering state must leave
    // both of them exactly where they were.
    var diag = {
      peerId: null,
      peersCreated: 0,
      offersSent: 0,
      answersApplied: 0,
      srcObjectAssignments: 0,
      trackIds: [],
      trackMutes: 0,
      trackUnmutes: 0,
      trackEnds: 0,
      videoEvents: {},
      events: []
    };

    function record(kind, detail) {
      var entry = { at: Date.now(), t: Math.round(performance.now()),
                    channel: channel, peer: diag.peerId, kind: kind,
                    detail: detail === undefined ? null : detail };
      diag.events.push(entry);
      if (diag.events.length > 240) diag.events.splice(0, diag.events.length - 240);
      if (global.DEMO_TRACE) {
        console.log("[src%d %s] %s %s", channel, diag.peerId || "-", kind,
                    detail === undefined ? "" : JSON.stringify(detail));
      }
      if (opts.onDiagnostic) opts.onDiagnostic(entry);
    }

    // The <video> element's own lifecycle, bound ONCE for the life of the
    // page.  If a command ever caused the element to be replaced or reloaded,
    // these counters would move - so they are the check for that, not a
    // narrative claim that it cannot happen.
    ["loadeddata", "playing", "waiting", "stalled", "emptied", "pause",
     "ended", "suspend", "error"].forEach(function (name) {
      video.addEventListener(name, function () {
        diag.videoEvents[name] = (diag.videoEvents[name] || 0) + 1;
        record("video." + name, { readyState: video.readyState });
      });
    });

    function report(patch) {
      Object.assign(status, patch || {});
      if (opts.onStatus) opts.onStatus(Object.assign({ channel: channel }, status));
    }

    function teardown() {
      if (pc) record("pc.close", { signalingState: pc.signalingState,
                                   connectionState: pc.connectionState });
      if (statsTimer !== null) { clearInterval(statsTimer); statsTimer = null; }
      if (frameHandle !== null && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(frameHandle);
        frameHandle = null;
      }
      if (rafHandle !== null) { cancelAnimationFrame(rafHandle); rafHandle = null; }
      if (dc) { try { dc.close(); } catch (e) {} dc = null; }
      if (pc) {
        pc.onconnectionstatechange = null;
        pc.ontrack = null;
        try { pc.close(); } catch (e) {}
        pc = null;
      }
      receiver = null;
    }

    function scheduleRetry(delay, why) {
      if (!live) return;
      status.reconnects += 1;
      record("retry.scheduled", { delayMs: delay, why: why });
      report({ phase: "waiting", text: why });
      if (retryTimer !== null) clearTimeout(retryTimer);
      retryTimer = setTimeout(function () { if (live) connect(); }, delay);
    }

    // One render step. `frameMeta` is present only via
    // requestVideoFrameCallback, which is where the frame's own RTP timestamp
    // comes from - and therefore where correct pairing comes from.
    function step(now, frameMeta) {
      if (!live) return;
      if (video.readyState < 2) {
        if (opts.onFrame) opts.onFrame({ ready: false, message: null, now: now });
        return;
      }
      // A presented frame is the only unambiguous evidence that media is
      // flowing: readyState stays >= 2 long after the pixels stop.
      media.lastFrameAt = Date.now();
      if (media.pauseUntil) {
        // The picture is back, so the declared window has served its purpose.
        record("pause.ended", { reason: media.pauseReason });
        media.pauseUntil = 0;
        media.pauseReason = null;
        report({ phase: "live", text: "live" });
      }
      if (!media.hadFrames) {
        media.hadFrames = true;
        report({ phase: "live", text: "live" });
      }
      presentation.callbacks += 1;
      if (frameMeta) {
        if (typeof frameMeta.presentedFrames === "number")
          presentation.presentedFrames = frameMeta.presentedFrames;
        if (typeof frameMeta.mediaTime === "number")
          presentation.mediaTime = frameMeta.mediaTime;
        // BROWSER RECEIVE -> SCREEN, per frame, from the browser's own
        // numbers.  `receiveTime` is when the last packet of this frame
        // arrived; `expectedDisplayTime` is when the compositor expects it on
        // screen.  Their difference is everything the browser adds: jitter
        // buffer residency + decode + compositing.  It is NOT the total
        // camera-to-screen latency - the sender-side measurement
        // (scripts/fps_forensics.py --pair) covers the ~1.08 s the video path
        // adds before the frame ever leaves the board.
        if (typeof frameMeta.receiveTime === "number" &&
            typeof frameMeta.expectedDisplayTime === "number") {
          presentation.browserMs =
            frameMeta.expectedDisplayTime - frameMeta.receiveTime;
          presentation.browserMsMax = Math.max(
            presentation.browserMsMax || 0, presentation.browserMs);
        }
        // Decode time alone, when the browser reports it.
        if (typeof frameMeta.processingDuration === "number") {
          presentation.decodeMs = frameMeta.processingDuration * 1000;
        }
      }
      var rtp = frameMeta ? frameMeta.rtpTimestamp : undefined;
      var hit = takeMetadata(store, rtp, opts.retentionMs || 0, now,
                             opts.holdMs || 0);
      if (hit && frameMeta && rtp !== undefined) latencySample(channel, frameMeta, hit.data, rtp);
      if (opts.onFrame) {
        opts.onFrame({ ready: true, message: hit ? hit.data : null, now: now,
                       rtpTimestamp: rtp, stats: store.stats });
      }
    }

    function pump() {
      if (!live) return;
      if (typeof video.requestVideoFrameCallback === "function") {
        frameHandle = video.requestVideoFrameCallback(function (now, meta) {
          step(now, meta);
          pump();
        });
        return;
      }
      // No requestVideoFrameCallback: fall back to rAF with no frame
      // timestamp, which is Insight's arrival-order path.
      rafHandle = requestAnimationFrame(function (now) {
        step(now, undefined);
        pump();
      });
    }

    function watchStats(mine) {
      statsTimer = setInterval(function () {
        if (!live || mine !== generation || !pc || pc.connectionState !== "connected") return;

        // A DECLARED pause: the application knows the source stopped on
        // purpose, so silence carries no information and nothing is measured,
        // probed or touched.  This is the branch that keeps a spoken command
        // from costing a panel when the board-resident voice server is in use.
        if (media.pauseUntil && Date.now() < media.pauseUntil) {
          if (status.phase !== "waiting") {
            report({ phase: "waiting",
                     text: media.pauseReason || "the source is paused" });
          }
          return;
        }

        // Silence check, before getStats: a displaced peer reports nothing
        // useful, so there is nothing in getStats to notice.
        var since = media.lastFrameAt || media.connectedAt || Date.now();
        var idle = Date.now() - since;
        var limit = media.hadFrames ? MEDIA_STALL_MS : MEDIA_NEVER_MS;
        if (idle > limit && !probing) {
          handleSilence(mine, idle);
          return;
        }
        if (!media.hadFrames && status.phase !== "waiting") {
          report({ phase: "connecting", text: "waiting for video" });
        }

        // Presentation rates, sampled on the same 1 s tick as getStats so
        // both numbers in the status line describe the same second.
        var nowMs = (typeof performance !== "undefined" && performance.now)
          ? performance.now() : Date.now();
        if (presentation.sampledAt !== null) {
          var presentDt = (nowMs - presentation.sampledAt) / 1000;
          if (presentDt > 0.2) {
            if (presentation.presentedFrames !== null &&
                presentation.lastPresentedSample !== null) {
              presentation.presentedFps = Math.max(0,
                (presentation.presentedFrames -
                 presentation.lastPresentedSample) / presentDt);
            }
            presentation.callbackFps =
              (presentation.callbacks - presentation.lastCallbackSample) /
              presentDt;
            if (presentation.mediaTime !== null &&
                presentation.lastMediaSample !== null) {
              presentation.mediaRate =
                (presentation.mediaTime - presentation.lastMediaSample) /
                presentDt;
            }
          }
        }
        presentation.lastPresentedSample = presentation.presentedFrames;
        presentation.lastCallbackSample = presentation.callbacks;
        presentation.lastMediaSample = presentation.mediaTime;
        presentation.sampledAt = nowMs;

        pc.getStats().then(function (report_) {
          var stalled = false;
          report_.forEach(function (entry) {
            if (entry.type !== "inbound-rtp" || entry.kind !== "video") return;
            var codec = null;
            if (entry.codecId) {
              var c = report_.get(entry.codecId);
              if (c && typeof c.mimeType === "string") {
                var name = (c.mimeType.split("/")[1] || "").toUpperCase();
                codec = name === "H264" ? "H.264" : (name === "H265" ? "H.265" : name);
              }
            }
            var kbps = null;
            if (meter.bytes !== null && meter.ts !== null && entry.timestamp > meter.ts) {
              kbps = Math.round(((entry.bytesReceived - meter.bytes) * 8) /
                                ((entry.timestamp - meter.ts) / 1000) / 1000);
            }
            meter.bytes = entry.bytesReceived;
            meter.ts = entry.timestamp;

            // viewer-react.js fp(): receiving but not decoding for a while is a
            // decoder stall and it reconnects. Kept, because it is the one case
            // where a reconnect is the correct answer rather than a regression.
            var receiving = typeof entry.framesReceived === "number" &&
                            decoderHealth.framesReceived !== null &&
                            entry.framesReceived > decoderHealth.framesReceived;
            var decoding = typeof entry.framesDecoded === "number" &&
                           decoderHealth.framesDecoded !== null &&
                           entry.framesDecoded > decoderHealth.framesDecoded;
            if (!decoding && receiving && typeof entry.framesDecoded === "number") {
              decoderHealth.stalledSince = decoderHealth.stalledSince || Date.now();
              if (Date.now() - decoderHealth.stalledSince > STALL_MS) stalled = true;
            } else {
              decoderHealth.stalledSince = null;
            }
            decoderHealth.framesReceived = entry.framesReceived;
            decoderHealth.framesDecoded = entry.framesDecoded;

            // THE "N ms buffer" NUMBER, precisely: the mean time a frame
            // spent in this browser's JITTER BUFFER, from "all its packets
            // have arrived" to "handed to the decoder".  W3C
            // RTCInboundRtpStreamStats.jitterBufferDelay is a running total
            // over jitterBufferEmittedCount frames, so the per-frame value is
            // the ratio of the two deltas.  It does NOT include capture, the
            // sender's pipeline, the network, decode, or compositing, and it
            // must never be read as the camera-to-screen latency.
            var bufferedMs = status.bufferedMs;
            if (typeof entry.jitterBufferDelay === "number" &&
                typeof entry.jitterBufferEmittedCount === "number") {
              if (jitter.delay !== null &&
                  entry.jitterBufferEmittedCount > jitter.count) {
                bufferedMs = Math.round(
                  ((entry.jitterBufferDelay - jitter.delay) * 1000) /
                  (entry.jitterBufferEmittedCount - jitter.count));
              }
              jitter.delay = entry.jitterBufferDelay;
              jitter.count = entry.jitterBufferEmittedCount;
            }

            report({
              phase: "live",
              text: "live",
              codec: codec || status.codec,
              width: entry.frameWidth || status.width,
              height: entry.frameHeight || status.height,
              fps: typeof entry.framesPerSecond === "number"
                ? Math.round(entry.framesPerSecond) : status.fps,
              kbps: kbps === null ? status.kbps : kbps,
              // What the browser is actually holding each frame for, and what
              // it was asked to hold it for.  The pair answers "is the lag
              // mine or the camera's": a low fps with a small buffer is the
              // sensor, a high fps with a large buffer is this setting.
              bufferedMs: bufferedMs,
              targetBufferMs: opts.syncBufferMs,
              // Stage 6.  Reported SEPARATELY from `fps` rather than replacing
              // it: the decoder rate is still true and still shown, and the
              // difference between the two is the finding.
              presentedFps: presentation.presentedFps === null
                ? null : Math.round(presentation.presentedFps * 10) / 10,
              // Everything the BROWSER adds, measured, kept separate from the
              // jitter-buffer figure below so neither can be mistaken for the
              // end-to-end latency.
              browserLatencyMs: presentation.browserMs === null
                ? null : Math.round(presentation.browserMs),
              browserLatencyMaxMs: presentation.browserMsMax === null
                ? null : Math.round(presentation.browserMsMax),
              decodeMs: presentation.decodeMs === null
                ? null : Math.round(presentation.decodeMs * 10) / 10,
              drawCallbackFps: presentation.callbackFps === null
                ? null : Math.round(presentation.callbackFps * 10) / 10,
              mediaRate: presentation.mediaRate === null
                ? null : Math.round(presentation.mediaRate * 100) / 100,
              framesDropped: typeof entry.framesDropped === "number"
                ? entry.framesDropped : status.framesDropped,
              messagesPerSec: meter.messages
            });
            meter.messages = 0;
          });
          if (stalled && live && mine === generation) {
            teardown();
            scheduleRetry(200, "decoder stalled; reconnecting");
          }
        }).catch(function () {});
      }, 1000);
    }

    // Silence has two causes and only one of them is ours to repair.
    //
    //   the source stopped sending      WAIT.  Touching the connection would
    //                                   be a reconnect that fixes nothing and
    //                                   delays the picture coming back.  The
    //                                   pipeline stops both cameras on purpose
    //                                   for every spoken command - measured at
    //                                   2.2-3.7 s, per channel, staggered.
    //   the source is sending and we    RE-OFFER.  vf gives a channel's media
    //   receive nothing                 to one peer, so we were displaced and
    //                                   offering again takes it back.
    //
    // Both look identical from here, so the server is asked: it can see the
    // UDP arriving at vf. Nothing is torn down until the answer says to.
    var probing = false;
    var whyConnect = "initial";

    function handleSilence(mine, idle) {
      // Never had a frame: there is no picture to lose, and re-offering is how
      // the first one arrives.  This is the initial-connect path.
      if (!media.hadFrames) {
        recover(mine, idle, "no media yet");
        return;
      }

      // We HAD a picture.  From here on, nothing is torn down without proof
      // that tearing it down is the repair.
      if (!opts.probeSource) {
        wait("waiting for the stream", { idleMs: idle, why: "no probe" });
        return;
      }
      probing = true;
      record("silence", { idleMs: idle, hadFrames: true });
      opts.probeSource(channel).then(function (info) {
        probing = false;
        if (!live || mine !== generation) return;

        if (info && info.live === false) {
          // Confirmed upstream: the source is not sending. Leave the media
          // session completely alone.
          status.sourcePauses += 1;
          record("source.not-sending", { bitrate: info.bitrate_bps || 0 });
          wait(media.hadFrames ? "camera paused at the source"
                               : "source is not sending",
               { idleMs: idle, why: "source paused" });
          return;
        }

        // The source IS sending and we are not presenting frames.  That is
        // displacement ONLY if something else is on the channel.  vf attaches
        // the video to one peer, so `peers > 1` is the whole evidence - and
        // with `peers <= 1` we ARE that peer, so re-offering would change
        // nothing except blanking the panel for a few seconds.
        var peers = info && typeof info.peers === "number" ? info.peers : null;
        if (peers === null) {
          // No evidence either way.  The conservative answer is to wait:
          // tearing down a working picture on no evidence is exactly what
          // produced the reported blackout, and a panel that waits recovers by
          // itself the moment frames return.
          record("source.live-but-silent", { peers: null, idleMs: idle,
                                             verdict: "vf did not report a peer "
                                                      + "count; waiting rather "
                                                      + "than guessing" });
          wait("waiting for the stream", { idleMs: idle, why: "no peer count" });
          return;
        }
        if (peers <= 1) {
          record("source.live-but-silent", { peers: peers, idleMs: idle,
                                             verdict: "we are the only peer; "
                                                      + "waiting, not re-offering" });
          wait("waiting for the stream", { idleMs: idle, peers: peers,
                                           why: "only peer" });
          return;
        }

        record("source.displaced", { peers: peers, idleMs: idle });
        recover(mine, idle, "displaced by another viewer");
      }).catch(function () {
        probing = false;
        // The probe itself failed; say so rather than reconnecting blind.
        record("probe.failed");
        wait("cannot reach the UI server", { idleMs: idle, why: "probe failed" });
      });
    }

    // Silence we have decided NOT to act on.  The clock is pushed forward so
    // the check does not re-fire every second, and the panel says what is
    // actually happening instead of pretending to be live.
    function wait(text, detail) {
      status.silenceWaits += 1;
      record("silence.waiting", detail || null);
      report({ phase: "waiting", text: text });
      media.lastFrameAt = Date.now();
    }

    // The ONLY place this module tears a working connection down on its own.
    // Reached from exactly two places: a connection that never had a frame
    // (nothing to lose) and proven displacement (re-offering is the repair).
    // An upstream camera pause reaches neither.
    function recover(mine, idle, reason) {
      if (!live || mine !== generation) return;
      var why = media.hadFrames
        ? "media was taken over; re-offering"
        : "waiting for the stream; retrying";
      if (media.hadFrames) status.mediaRecoveries += 1;
      record("recover", { reason: reason, idleMs: idle,
                          hadFrames: media.hadFrames });
      media.hadFrames = false;
      teardown();
      scheduleRetry(300, why);
    }

    function connect() {
      if (!live) return;
      teardown();
      generation += 1;
      var mine = generation;
      store = newStore();
      decoderHealth = { framesReceived: null, framesDecoded: null, stalledSince: null };
      meter = { bytes: null, ts: null, messages: 0, codec: null };
      // The declared-pause window and its counter SURVIVE a reconnect: a pause
      // the application declared is still true across one, and losing it here
      // would re-arm the silence detector inside the very window it was told
      // to ignore.
      media = { lastFrameAt: 0, connectedAt: Date.now(), hadFrames: false,
                pauseUntil: media.pauseUntil, pauseReason: media.pauseReason,
                declaredPauses: media.declaredPauses };
      report({ phase: "connecting", text: "connecting" });

      diag.peersCreated += 1;
      diag.peerId = "p" + diag.peersCreated;
      record("pc.create", { peersCreated: diag.peersCreated, why: whyConnect });
      whyConnect = "retry";

      pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
      receiver = pc.addTransceiver("video", { direction: "recvonly" }).receiver;
      pc.oniceconnectionstatechange = function () {
        if (pc) record("iceConnectionState", pc.iceConnectionState);
      };
      pc.onicegatheringstatechange = function () {
        if (pc) record("iceGatheringState", pc.iceGatheringState);
      };
      pc.onsignalingstatechange = function () {
        if (pc) record("signalingState", pc.signalingState);
      };
      if (receiver && "jitterBufferTarget" in receiver) {
        try { receiver.jitterBufferTarget = opts.syncBufferMs; } catch (e) {}
      }
      dc = pc.createDataChannel("metadata", { ordered: true, maxRetransmits: 0 });
      dc.onmessage = function (event) {
        if (!live || mine !== generation) return;
        try {
          storeMetadata(store, JSON.parse(event.data), performance.now());
          meter.messages += 1;
        } catch (e) { /* a malformed message is dropped, never fatal */ }
      };
      pc.ontrack = function (event) {
        if (!live || mine !== generation || event.track.kind !== "video") return;
        var track = event.track;
        diag.trackIds.push(track.id);
        record("track", { id: track.id, kind: track.kind,
                          readyState: track.readyState, muted: track.muted });
        track.onmute = function () {
          diag.trackMutes += 1;
          record("track.mute", { id: track.id });
        };
        track.onunmute = function () {
          diag.trackUnmutes += 1;
          record("track.unmute", { id: track.id });
        };
        track.onended = function () {
          diag.trackEnds += 1;
          record("track.ended", { id: track.id });
        };
        var stream = new MediaStream();
        stream.addTrack(track);
        diag.srcObjectAssignments += 1;
        record("srcObject.set", { assignments: diag.srcObjectAssignments,
                                  streamId: stream.id });
        video.srcObject = stream;
        var play = video.play();
        if (play && play.catch) play.catch(function () {});
        // A track, not yet a picture. "live" is claimed only once a frame has
        // actually been presented - see step(). Saying "live" here is what made
        // a black panel read as a working one.
        report({ phase: "connecting", text: "waiting for video" });
      };
      pc.onconnectionstatechange = function () {
        if (!pc || !live || mine !== generation) return;
        var state = pc.connectionState;
        record("connectionState", state);
        if (state === "failed" || state === "disconnected" || state === "closed") {
          teardown();
          scheduleRetry(RETRY_FAIL_MS, "connection " + state + "; retrying");
        }
      };

      watchStats(mine);
      pump();

      diag.offersSent += 1;
      record("offer.post", { offersSent: diag.offersSent });
      negotiate(pc, "/offer?channel=" + channel).then(function (answer) {
        if (!live || mine !== generation) return;
        diag.answersApplied += 1;
        record("answer.apply", { answersApplied: diag.answersApplied });
        return pc.setRemoteDescription(answer);
      }).catch(function (error) {
        if (!live || mine !== generation) return;
        teardown();
        if (error && error.status === 415) {
          // Permanent for this browser. Insight's own viewer never retries it.
          report({ phase: "error",
                   text: "this browser cannot decode the stream's codec" });
          return;
        }
        if (retryable(error && error.status)) {
          scheduleRetry(RETRY_MS, "waiting for the stream");
          return;
        }
        report({ phase: "error",
                 text: "connection failed: " + ((error && error.message) || "unknown") });
      });
    }

    connect();

    return {
      channel: channel,

      /**
       * "The source is about to stop on purpose; do not interpret the silence."
       *
       * Called by the code that CAUSES the pause - the only place that can
       * know.  With the board-resident voice server a spoken command stops both
       * cameras for the Whisper/Qwen window plus a staggered restore, so the
       * page declares a window before it sends the audio and this source then
       * measures nothing, probes nothing and touches nothing until a frame
       * arrives or the window expires.
       *
       * It can only ever SUPPRESS a reconnect, never cause one.
       */
      expectSourcePause: function (reason, ms) {
        var until = Date.now() + Math.max(0, ms || 0);
        if (until <= media.pauseUntil) return;
        media.pauseUntil = until;
        media.pauseReason = reason || "the source is paused";
        media.declaredPauses += 1;
        status.declaredPauses = media.declaredPauses;
        record("pause.declared", { reason: media.pauseReason, ms: ms });
      },

      /**
       * The media lifecycle, for the identity checks: is this still the same
       * <video> node, the same MediaStream, the same track and the same peer
       * connection it was before?  Everything here is an identity, not a
       * measurement, so a caller can compare two snapshots for equality.
       */
      lifecycle: function () {
        var stream = video.srcObject || null;
        var tracks = stream && stream.getTracks ? stream.getTracks() : [];
        return {
          channel: channel,
          peerId: diag.peerId,
          peersCreated: diag.peersCreated,
          offersSent: diag.offersSent,
          srcObjectAssignments: diag.srcObjectAssignments,
          streamId: stream ? stream.id : null,
          trackId: tracks.length ? tracks[0].id : null,
          trackReadyState: tracks.length ? tracks[0].readyState : null,
          videoNodeId: video.__demoNodeId || null,
          videoReadyState: video.readyState,
          connectionState: pc ? pc.connectionState : null,
          declaredPauses: media.declaredPauses,
          pauseActive: !!(media.pauseUntil && Date.now() < media.pauseUntil),
          reconnects: status.reconnects,
          mediaRecoveries: status.mediaRecoveries,
          silenceWaits: status.silenceWaits
        };
      },

      stats: function () { return store.stats; },
      status: function () { return Object.assign({}, status); },
      diagnostics: function () {
        return {
          channel: channel,
          peerId: diag.peerId,
          peersCreated: diag.peersCreated,
          offersSent: diag.offersSent,
          answersApplied: diag.answersApplied,
          srcObjectAssignments: diag.srcObjectAssignments,
          trackIds: diag.trackIds.slice(),
          trackMutes: diag.trackMutes,
          trackUnmutes: diag.trackUnmutes,
          trackEnds: diag.trackEnds,
          videoEvents: Object.assign({}, diag.videoEvents),
          reconnects: status.reconnects,
          mediaRecoveries: status.mediaRecoveries,
          sourcePauses: status.sourcePauses,
          connectionState: pc ? pc.connectionState : null,
          iceConnectionState: pc ? pc.iceConnectionState : null,
          signalingState: pc ? pc.signalingState : null,
          videoReadyState: video.readyState,
          events: diag.events.slice(-60)
        };
      },
      stop: function () {
        live = false;
        if (retryTimer !== null) { clearTimeout(retryTimer); retryTimer = null; }
        teardown();
      }
    };
  }

  global.InsightSource = {
    open: openSource,
    MAX_PENDING: MAX_PENDING,
    MEDIA_STALL_MS: MEDIA_STALL_MS,
    MEDIA_NEVER_MS: MEDIA_NEVER_MS,
    // Exposed for scripts/render_test.js, the way Insight's own drawing.js
    // exposes window.decodeRleMaskAlpha for its rleMask test. The pairing rule
    // is the thing most worth testing and it is not reachable through open().
    _internals: {
      newStore: newStore,
      storeMetadata: storeMetadata,
      takeMetadata: takeMetadata,
      expire: expire
    }
  };
})(window);
