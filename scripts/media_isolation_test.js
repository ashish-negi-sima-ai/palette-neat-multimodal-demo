// media_isolation_test.js - a command must not touch the media session.
//
//     node scripts/media_isolation_test.js
//
// Runs web/webrtc.js in a fake DOM with a fake RTCPeerConnection that COUNTS
// everything: constructions, offers, setRemoteDescription, srcObject writes,
// close(). Then it drives exactly the sequence the spec's success criteria
// name - All -> Person -> Dog -> All, and Detection ON -> OFF -> ON, several
// times over - and asserts the counters have not moved.
//
// This is the check that cannot be made by reading the code, because the claim
// is about what does NOT happen. The filter lives in overlay.js and is read
// per drawn frame; there is no wiring from it to the connection at all, and
// this proves that empirically rather than by assertion.

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const APP_DIR = path.dirname(__dirname);
const failures = [];

function check(label, condition, extra) {
  console.log(`  [${condition ? "PASS" : "FAIL"}] ${label}` +
    (condition ? "" : `\n         -> ${JSON.stringify(extra)}`));
  if (!condition) failures.push(label);
}

// ----------------------------------------------------------- the fake world

function buildSandbox(counters) {
  const listeners = {};

  class FakeTrack {
    constructor(id) {
      this.id = id; this.kind = "video"; this.readyState = "live"; this.muted = false;
    }
    stop() { counters.trackStops += 1; }
  }

  class FakeMediaStream {
    constructor() { this.id = "stream" + (++counters.streams); this.tracks = []; }
    addTrack(track) { this.tracks.push(track); }
    getTracks() { return this.tracks; }
  }

  class FakeRTCPeerConnection {
    constructor() {
      counters.pcCreated += 1;
      this.connectionState = "new";
      this.iceConnectionState = "new";
      this.iceGatheringState = "new";
      this.signalingState = "stable";
      this.ontrack = null;
      this.onconnectionstatechange = null;
      this.closed = false;
    }
    addTransceiver() {
      return { receiver: { jitterBufferTarget: null, track: null } };
    }
    createDataChannel(label) {
      counters.dataChannels += 1;
      const dc = { label, onmessage: null, close() { counters.dcClosed += 1; } };
      this.dc = dc;
      // The test feeds metadata through this, so it needs a handle on it.
      counters.dc = dc;
      return dc;
    }
    createOffer() {
      counters.createOffer += 1;
      return Promise.resolve({ type: "offer", sdp: "v=0\r\n" });
    }
    setLocalDescription() { return Promise.resolve(); }
    setRemoteDescription() {
      counters.setRemoteDescription += 1;
      // Deliver a track, exactly as a real answer does.
      const self = this;
      return Promise.resolve().then(function () {
        if (self.ontrack) {
          self.ontrack({ track: new FakeTrack("track" + (++counters.tracks)) });
        }
        self.connectionState = "connected";
        self.iceConnectionState = "connected";
        if (self.onconnectionstatechange) self.onconnectionstatechange();
      });
    }
    getStats() { return Promise.resolve(new Map()); }
    close() { counters.pcClosed += 1; this.closed = true; }
  }

  let videoSrc = null;
  const video = {
    videoWidth: 1280, videoHeight: 720, readyState: 4,
    get srcObject() { return videoSrc; },
    set srcObject(value) { counters.srcObjectWrites += 1; videoSrc = value; },
    play() { counters.play += 1; return Promise.resolve(); },
    addEventListener(name, fn) { (listeners[name] = listeners[name] || []).push(fn); },
    // The render loop. Driven manually by the test so frames are deterministic.
    requestVideoFrameCallback(fn) {
      counters.rvfc += 1;
      video._pending = fn;
      return counters.rvfc;
    },
    cancelVideoFrameCallback() { video._pending = null; },
    _pending: null
  };

  const sandbox = {
    console,
    performance: { now: () => Date.now() },
    Date, Number, Math, JSON, Object, Array, String, Boolean, Map, Set,
    Promise, Error, Uint8ClampedArray, Infinity,
    RTCPeerConnection: FakeRTCPeerConnection,
    MediaStream: FakeMediaStream,
    setInterval: (fn, ms) => { sandbox._intervals.push(fn); return sandbox._intervals.length; },
    clearInterval: () => {},
    setTimeout: (fn, ms) => { sandbox._timeouts.push(fn); return sandbox._timeouts.length; },
    clearTimeout: () => {},
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
    fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve({ type: "answer", sdp: "v=0\r\n" }) }),
    localStorage: { getItem: () => null, setItem: () => undefined },
    ImageData: class {},
    document: { createElement: () => ({ width: 0, height: 0, getContext: () => ({}) }) },
    _intervals: [],
    _timeouts: [],
    _video: video,
    _listeners: listeners
  };
  sandbox.window = sandbox;
  sandbox.global = sandbox;
  return sandbox;
}

function makeCanvas(width, height) {
  const calls = [];
  const ctx = {
    canvas: null, calls,
    save() {}, restore() {},
    clearRect() { calls.push("clearRect"); },
    fillRect() {}, strokeRect() { calls.push("strokeRect"); },
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {},
    fill() {}, stroke() {}, fillText() {}, setLineDash() {},
    measureText: (t) => ({ width: t.length * 7 }),
    drawImage() {}, putImageData() {},
    strokeStyle: null, fillStyle: null, lineWidth: null, font: null,
    globalAlpha: 1, globalCompositeOperation: "source-over",
    imageSmoothingEnabled: true
  };
  const canvas = { width, height, clientWidth: width, clientHeight: height,
                   getContext: () => ctx };
  ctx.canvas = canvas;
  return canvas;
}

const MESSAGE = {
  type: "object-detection", timestamp: 1000, frame_id: "1",
  _insight: { rtp_timestamp: 90000 },
  data: { objects: [
    { id: "0", label: "person", confidence: 0.9, bbox: [10, 10, 40, 80] },
    { id: "1", label: "dog", confidence: 0.8, bbox: [90, 20, 30, 30] },
    { id: "2", label: "car", confidence: 0.7, bbox: [200, 40, 60, 40] }
  ] }
};

const RENDER = { video_sync_buffer_ms: 350, metadata_retention_ms: 2000,
                 metadata_hold_ms: 150, mask_opacity: 0.45,
                 confidence_threshold: 0, box_width: 2 };

function run() {
  const counters = {
    pcCreated: 0, pcClosed: 0, createOffer: 0, setRemoteDescription: 0,
    srcObjectWrites: 0, dataChannels: 0, dcClosed: 0, tracks: 0, streams: 0,
    trackStops: 0, play: 0, rvfc: 0
  };
  const sandbox = buildSandbox(counters);
  const context = vm.createContext(sandbox);

  // Insight's renderer and our filter, so the whole draw path is exercised.
  const drawingPath = path.join(APP_DIR, "runtime", "cache", "drawing.js");
  if (fs.existsSync(drawingPath)) {
    vm.runInContext(fs.readFileSync(drawingPath, "utf8"), context,
                    { filename: "drawing.js" });
  } else {
    // The renderer is not required for this test; a stub keeps it honest
    // about which strategies were asked for.
    vm.runInContext("window.drawStrategies = { 'object-detection': function(ctx){ ctx.strokeRect(0,0,1,1); } };",
                    context, { filename: "stub-drawing.js" });
  }
  vm.runInContext(fs.readFileSync(path.join(APP_DIR, "web", "overlay.js"), "utf8"),
                  context, { filename: "overlay.js" });
  vm.runInContext(fs.readFileSync(path.join(APP_DIR, "web", "webrtc.js"), "utf8"),
                  context, { filename: "webrtc.js" });

  const video = sandbox._video;
  const canvas = makeCanvas(640, 360);
  const ctx = canvas.getContext("2d");

  // The panel's render state - exactly the shape app.js keeps.
  let plan = { detection_enabled: true, classes: null, position: "left" };
  let drawn = { kept: 0, total: 0 };

  const conn = sandbox.InsightSource.open({
    channel: 0,
    video: video,
    syncBufferMs: RENDER.video_sync_buffer_ms,
    retentionMs: RENDER.metadata_retention_ms,
    holdMs: RENDER.metadata_hold_ms,
    probeSource: () => Promise.resolve({ live: true, bitrate_bps: 4000000 }),
    onStatus: () => {},
    onFrame: function (frame) {
      if (!frame.ready) return;
      drawn = sandbox.Overlay.draw(ctx, canvas, video, 0, frame.message,
                                   plan, RENDER);
    }
  });

  // Let the fake negotiation settle, then present frames on demand.
  function settle() { return new Promise((r) => setImmediate(r)); }

  function presentFrame(rtpTimestamp) {
    // metadata first, as it arrives ahead of the frame it describes
    const message = JSON.parse(JSON.stringify(MESSAGE));
    message._insight.rtp_timestamp = rtpTimestamp;
    if (counters.dc && counters.dc.onmessage) {
      counters.dc.onmessage({ data: JSON.stringify(message) });
    }
    const pending = video._pending;
    video._pending = null;
    if (pending) pending(Date.now(), { rtpTimestamp: rtpTimestamp });
  }

  return settle().then(settle).then(function () {
    check("one RTCPeerConnection after connect", counters.pcCreated === 1,
          counters);
    check("one offer after connect", counters.createOffer === 1, counters);
    check("one answer applied", counters.setRemoteDescription === 1, counters);
    check("one srcObject assignment", counters.srcObjectWrites === 1, counters);
    check("one data channel", counters.dataChannels === 1, counters);

    const baseline = Object.assign({}, counters);
    const diagBefore = conn.diagnostics();

    // ---- the success criteria, several times over --------------------
    const FILTERS = [null, "person", "dog", null, "person", "dog", null];
    const DETECTION = [true, false, true, false, true];
    let frame = 0;

    for (let round = 0; round < 3; round += 1) {
      FILTERS.forEach(function (name) {
        plan = { detection_enabled: true, classes: [name], position: "left" };
        presentFrame(90000 + (frame += 1) * 3000);
      });
      DETECTION.forEach(function (enabled) {
        plan = { detection_enabled: enabled, classes: ["person"],
                 position: "left" };
        presentFrame(90000 + (frame += 1) * 3000);
      });
      // a swap: the panel this source is shown in changes, nothing else
      plan = { detection_enabled: true, classes: ["person"],
               position: round % 2 ? "left" : "right" };
      presentFrame(90000 + (frame += 1) * 3000);
    }

    console.log(`\n  drove ${frame} frames across ${3 * (FILTERS.length + DETECTION.length + 1)} state changes\n`);

    check("NO new RTCPeerConnection was created",
          counters.pcCreated === baseline.pcCreated,
          { before: baseline.pcCreated, after: counters.pcCreated });
    check("NO new /offer was generated",
          counters.createOffer === baseline.createOffer,
          { before: baseline.createOffer, after: counters.createOffer });
    check("NO renegotiation (setRemoteDescription unchanged)",
          counters.setRemoteDescription === baseline.setRemoteDescription,
          counters);
    check("srcObject was never reassigned",
          counters.srcObjectWrites === baseline.srcObjectWrites,
          { before: baseline.srcObjectWrites, after: counters.srcObjectWrites });
    check("no RTCPeerConnection was closed",
          counters.pcClosed === 0, counters);
    check("no data channel was closed",
          counters.dcClosed === 0, counters);
    check("no track was stopped", counters.trackStops === 0, counters);
    check("no second track appeared", counters.tracks === 1, counters);

    const diagAfter = conn.diagnostics();
    check("the connection's own counters agree",
          diagAfter.peersCreated === diagBefore.peersCreated &&
          diagAfter.offersSent === diagBefore.offersSent &&
          diagAfter.srcObjectAssignments === diagBefore.srcObjectAssignments,
          { before: diagBefore, after: diagAfter });
    check("peers created per stream is exactly 1",
          diagAfter.peersCreated === 1, diagAfter.peersCreated);
    check("offers sent per stream is exactly 1",
          diagAfter.offersSent === 1, diagAfter.offersSent);
    check("no track mute, unmute or end occurred",
          diagAfter.trackMutes === 0 && diagAfter.trackUnmutes === 0 &&
          diagAfter.trackEnds === 0, diagAfter);
    check("no media recovery or reconnect was triggered",
          diagAfter.reconnects === 0 && diagAfter.mediaRecoveries === 0,
          diagAfter);

    // And the rendering really did follow the state, so the test is not
    // passing merely because nothing happened at all.
    plan = { detection_enabled: true, classes: ["dog"], position: "left" };
    presentFrame(90000 + (frame += 1) * 3000);
    check("the filter still takes effect (1 of 3 drawn)",
          drawn.kept === 1 && drawn.total === 3, drawn);
    plan = { detection_enabled: false, classes: ["dog"], position: "left" };
    presentFrame(90000 + (frame += 1) * 3000);
    check("detection OFF still suppresses drawing (0 of 3)",
          drawn.kept === 0 && drawn.total === 3, drawn);
    plan = { detection_enabled: true, classes: null, position: "left" };
    presentFrame(90000 + (frame += 1) * 3000);
    check("All still draws everything (3 of 3)",
          drawn.kept === 3 && drawn.total === 3, drawn);

    console.log("");
    if (failures.length) {
      console.log(`${failures.length} check(s) FAILED`);
      failures.forEach((l) => console.log("  - " + l));
      process.exitCode = 1;
      return;
    }
    console.log("all checks passed");
  });
}

console.log("media isolation test: a command must not touch the media session\n");
run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
