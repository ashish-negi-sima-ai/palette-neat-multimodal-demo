// app.js - the Palette NEAT SDK Demo page.
//
// 1. A SOURCE owns a <video>, a <canvas> and an RTCPeerConnection for the whole
//    life of the page. Nothing in this file closes, reopens or re-offers one in
//    response to a command.
// 2. A camera view is one unit: its video, metadata, overlay, label, status AND its
//    overlay state (detection_enabled, classes, box_color, kept per source by the
//    server). The server decides which source is on which side; this file sets one
//    CSS custom property per card. That is the swap.
// 3. Drawing is Insight's: window.drawStrategies comes from the installed
//    Insight's drawing.js; overlay.js only filters what reaches it.
// 4. Voice and typed text reach the SAME server-side command path
//    (backend/command_processor.py). This page only shows its result.

(function () {
  "use strict";

  var POSITIONS = ["left", "right"];
  var SCALES = [1, 1.25, 1.5];
  var query = new URLSearchParams(window.location.search);
  // Diagnostics only: ?overlay=off draws no canvas, ?only=0|1 opens one source.
  var OVERLAY_ENABLED = query.get("overlay") !== "off";
  var ONLY_SOURCE = query.get("only");

  var cfg = null;
  var state = null;
  var lastRecord = null;
  var voiceState = null;
  var sources = {};
  var diagTimer = null;
  var qwenFlashTimer = null;
  var pendingLocal = false;

  var mediaLog = {
    entries: [],
    push: function (entry) {
      this.entries.push(entry);
      if (this.entries.length > 400) this.entries.splice(0, this.entries.length - 400);
    },
    note: function (kind, detail) {
      this.push({ at: Date.now(), channel: null, kind: kind, detail: detail });
    },
    text: function (limit) {
      return this.entries.slice(-(limit || 30)).map(function (e) {
        var when = new Date(e.at).toISOString().substr(11, 12);
        var who = e.channel === null || e.channel === undefined ? "ui  " : ("src" + e.channel);
        return when + "  " + who + "  " + e.kind + (e.detail ? "  " + JSON.stringify(e.detail) : "");
      }).join("\n");
    }
  };
  window.demoMediaLog = mediaLog;

  var frameRates = {};
  window.demoFrameRates = function () {
    var out = { at: new Date().toISOString(), sources: {} };
    Object.keys(sources).forEach(function (key) {
      var entry = sources[key];
      var life = entry.conn && entry.conn.lifecycle ? entry.conn.lifecycle() : {};
      out.sources[key] = {
        video: frameRates[key] || null,
        metadataPerSec: entry.metaRate || null,
        reconnects: life.reconnects,
        mediaRecoveries: life.mediaRecoveries,
        peersCreated: life.peersCreated,
        connectionState: life.connectionState,
        videoReadyState: life.videoReadyState
      };
    });
    return out;
  };

  // One entry per camera card: which channel the card's video and metadata come
  // from, where the card is on screen, and the overlay plan it is drawn with.
  //   copy(JSON.stringify(window.demoViews(), null, 1))
  window.demoViews = function () {
    return Object.keys(sources).map(function (key) {
      var entry = sources[key];
      var rect = entry.card.getBoundingClientRect();
      return {
        source: key, channel: entry.channel, task: cfg.sources[key].task,
        side: entry.side.textContent, model: entry.model.textContent,
        mode: entry.mode.textContent, column: getComputedStyle(entry.card).gridColumnStart,
        left: Math.round(rect.left), counts: entry.lastCounts,
        meta: entry.lastMeta || null, plan: planForSource(key),
        pairing: entry.conn.stats()
      };
    });
  };

  function $(id) { return document.getElementById(id); }
  var el = {
    statusCameras: $("statusCameras"),
    statusVoice: $("statusVoice"),
    controlsToggle: $("controlsToggle"),
    sheet: $("sheet"),
    scrim: $("scrim"),
    sheetClose: $("sheetClose"),
    inputLabel: $("inputLabel"),
    inputText: $("inputText"),
    commandText: $("commandText"),
    talk: $("talkBtn"),
    talkLabel: $("talkLabel"),
    languageSeg: $("languageSeg"),
    fontSeg: $("fontSeg"),
    textForm: $("textForm"),
    textInput: $("textInput"),
    textRun: $("textRun"),
    clearBtn: $("clearBtn"),
    manual: $("manualControls"),
    swap: $("swapBtn"),
    commandOut: $("commandOut"),
    diagOut: $("diagOut"),
    cameraOut: $("cameraOut"),
    wl: { det: $("wl-det"), seg: $("wl-seg"), whisper: $("wl-whisper"), qwen: $("wl-qwen") }
  };

  // ---------------------------------------------------------------- helpers

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { return null; }
    return null;
  }

  function postJSON(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        return data;
      });
    });
  }

  function titleCase(text) {
    return String(text || "").charAt(0).toUpperCase() + String(text || "").slice(1);
  }

  function setMeta(chip, text) {
    var meta = chip.querySelector(".wl-meta");
    if (meta.textContent !== text) meta.textContent = text;
  }

  // A view's overlay state belongs to its source (the same channel as its video
  // and metadata), so it is looked up by source and moves with the card on a swap.
  function planForSource(key) {
    if (state && state.sources && state.sources[key]) {
      var view = state.sources[key];
      return { position: view.position, detection_enabled: view.detection_enabled,
               classes: view.classes, box_color: view.box_color };
    }
    if (state) {
      for (var i = 0; i < POSITIONS.length; i += 1) {
        var position = POSITIONS[i];
        if (String(state.mapping[position]) === String(key)) {
          var entry = state.positions[position];
          return { position: position, detection_enabled: entry.detection_enabled,
                   classes: entry.classes, box_color: entry.box_color };
        }
      }
    }
    return { detection_enabled: true, classes: null, box_color: null, position: null };
  }

  // ------------------------------------------------------------- font size

  function applyScale(scale) {
    if (SCALES.indexOf(scale) < 0) scale = 1;
    document.documentElement.style.fontSize = (16 * scale) + "px";
    document.documentElement.classList.toggle("scaled", scale > 1);
    Array.prototype.forEach.call(el.fontSeg.querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(parseFloat(b.getAttribute("data-scale")) === scale));
    });
    store("demo.fontScale", String(scale));
  }

  function wireFontSize() {
    applyScale(parseFloat(store("demo.fontScale") || "1"));
    el.fontSeg.addEventListener("click", function (event) {
      var scale = event.target.getAttribute && event.target.getAttribute("data-scale");
      if (scale) applyScale(parseFloat(scale));
    });
  }

  // -------------------------------------------------------------- language

  // Every page load starts in the configured default language
  // (config/ui_config.json voice.language, KO). A choice made on the page lasts for
  // this page only, so a browser left on Auto or EN still opens in KO next time.
  var selectedLanguage = null;

  function currentLanguage() {
    var allowed = (cfg && cfg.voice && cfg.voice.languages) || ["auto", "ko", "en"];
    if (selectedLanguage && allowed.indexOf(selectedLanguage) >= 0) return selectedLanguage;
    return (cfg && cfg.voice && cfg.voice.default_language) || "ko";
  }

  function renderLanguage() {
    var lang = currentLanguage();
    Array.prototype.forEach.call(el.languageSeg.querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-lang") === lang));
    });
  }

  function wireLanguage() {
    try { localStorage.removeItem("demo.language"); } catch (e) {}
    renderLanguage();
    el.languageSeg.addEventListener("click", function (event) {
      var lang = event.target.getAttribute && event.target.getAttribute("data-lang");
      if (!lang) return;
      selectedLanguage = lang;
      renderLanguage();
      refreshReadout();
      mediaLog.note("language", { language: lang });
    });
  }

  // ------------------------------------------------------------- the sources

  function modeText(entry) {
    var bits = [];
    if (cfg.capture && cfg.capture.label) bits.push(cfg.capture.label);
    var codec = entry.lastStatus && entry.lastStatus.codec;
    if (codec) bits.push(codec);
    return bits.join(" | ");
  }

  function buildSource(key) {
    var card = $("cam-" + key);
    var entry = {
      key: key,
      channel: cfg.sources[key].channel,
      card: card,
      side: card.querySelector('[data-role="side"]'),
      model: card.querySelector('[data-role="model"]'),
      mode: card.querySelector('[data-role="mode"]'),
      video: card.querySelector('[data-role="video"]'),
      canvas: card.querySelector('[data-role="canvas"]'),
      badge: card.querySelector('[data-role="badge"]'),
      detection: card.querySelector('[data-role="detection"]'),
      classes: card.querySelector('[data-role="classes"]'),
      boxcolor: card.querySelector('[data-role="boxcolor"]'),
      swatch: card.querySelector('[data-role="swatch"]'),
      counts: card.querySelector('[data-role="counts"]'),
      lastCounts: "",
      lastStatus: null
    };
    entry.ctx = entry.canvas.getContext("2d");
    entry.video.__demoNodeId = "video-" + key;
    entry.canvas.__demoNodeId = "canvas-" + key;
    entry.model.textContent = cfg.sources[key].model || cfg.sources[key].detail;

    entry.conn = window.InsightSource.open({
      channel: entry.channel,
      video: entry.video,
      syncBufferMs: cfg.render.video_sync_buffer_ms,
      retentionMs: cfg.render.metadata_retention_ms,
      holdMs: cfg.render.metadata_hold_ms,
      onStatus: function (status) { renderStatus(entry, status); },
      probeSource: function (ch) {
        return fetch("/api/sources/live", { cache: "no-store" })
          .then(function (r) { return r.json(); })
          .then(function (data) {
            var found = null;
            Object.keys(data.sources || {}).forEach(function (k) {
              if (data.sources[k].channel === ch) found = data.sources[k];
            });
            return found;
          });
      },
      onDiagnostic: function (event) { mediaLog.push(event); },
      onFrame: function (frame) {
        if (!frame.ready) return;
        var t0 = performance.now();
        var result = OVERLAY_ENABLED
          ? window.Overlay.draw(entry.ctx, entry.canvas, entry.video, entry.channel,
                                frame.message, planForSource(key), cfg.render)
          : window.Overlay.filterMessage(frame.message, planForSource(key));
        // What this card's picture was drawn with, for window.demoViews().
        if (frame.message) {
          var listKey = window.Overlay.LIST_KEY[frame.message.type];
          var items = listKey && frame.message.data ? frame.message.data[listKey] : null;
          var insightRtp = frame.message._insight ? frame.message._insight.rtp_timestamp : undefined;
          entry.lastMeta = {
            type: frame.message.type,
            paired: frame.rtpTimestamp !== undefined && insightRtp !== undefined &&
                     (insightRtp >>> 0) === (frame.rtpTimestamp >>> 0),
            labels: Array.isArray(items) ? items.map(function (e) { return e.label; }) : null,
            at: Date.now()
          };
        }
        if (frame.stats) {
          var nowMs = performance.now();
          if (entry.metaSampledAt === undefined) {
            entry.metaSampledAt = nowMs;
            entry.metaBase = { received: frame.stats.received, matches: frame.stats.matches };
          } else if (nowMs - entry.metaSampledAt >= 1000) {
            var seconds = (nowMs - entry.metaSampledAt) / 1000;
            entry.metaRate = {
              receivedPerSec: Math.round((frame.stats.received - entry.metaBase.received) / seconds * 10) / 10,
              matchedPerSec: Math.round((frame.stats.matches - entry.metaBase.matches) / seconds * 10) / 10
            };
            entry.metaSampledAt = nowMs;
            entry.metaBase = { received: frame.stats.received, matches: frame.stats.matches };
          }
        }
        var cost = performance.now() - t0;
        entry.drawMs = entry.drawMs === undefined ? cost : (entry.drawMs * 0.9 + cost * 0.1);
        renderCounts(entry, result);
      }
    });

    sources[key] = entry;
    return entry;
  }

  function renderStatus(entry, status) {
    entry.lastStatus = status;
    var live = status.phase === "live" && entry.video.readyState >= 2;
    entry.badge.hidden = live;
    if (!live) {
      entry.badge.textContent = titleCase(status.text);
      entry.badge.className = "cam-badge" +
        (status.phase === "error" ? " bad" : status.phase === "waiting" ? " warn" : "");
    }
    var text = modeText(entry);
    entry.mode.hidden = !text;
    if (entry.mode.textContent !== text) entry.mode.textContent = text;
    frameRates[entry.key] = {
      decodedFps: status.fps, presentedFps: status.presentedFps, codec: status.codec,
      width: status.width, height: status.height, kbps: status.kbps,
      jitterBufferMs: status.bufferedMs, jitterBufferTargetMs: status.targetBufferMs,
      browserLatencyMs: status.browserLatencyMs, framesDropped: status.framesDropped,
      overlayMsAvg: entry.drawMs === undefined ? null : Math.round(entry.drawMs * 10) / 10
    };
    renderCameraStatus();
  }

  function renderCameraStatus() {
    var keys = Object.keys(sources);
    var live = keys.filter(function (k) {
      var s = sources[k].lastStatus;
      return s && s.phase === "live" && sources[k].video.readyState >= 2;
    }).length;
    var label = "Cameras " + live + "/" + keys.length;
    var node = el.statusCameras.querySelector("span");
    if (node.textContent !== label) node.textContent = label;
    el.statusCameras.className = "status " + (live === keys.length && keys.length ? "ok" : (live ? "warn" : "bad"));
  }

  function renderCounts(entry, result) {
    var text;
    if (!result || (result.total === 0 && result.kept === 0)) text = "0 objects";
    else if (result.kept === result.total) text = result.total + (result.total === 1 ? " object" : " objects");
    else text = result.kept + " of " + result.total + " objects";
    if (text !== entry.lastCounts) {
      entry.lastCounts = text;
      entry.counts.textContent = text;
    }
  }

  // -------------------------------------------------------------- the layout

  function applyState(next) {
    state = next;
    POSITIONS.forEach(function (position, index) {
      var key = String(state.mapping[position]);
      var entry = sources[key];
      if (!entry) return;
      entry.card.style.setProperty("--col", String(index + 1));
      entry.side.textContent = position.toUpperCase();

      var panel = state.positions[position];
      entry.detection.textContent = panel.detection_enabled ? "On" : "Off";
      entry.detection.className = panel.detection_enabled ? "" : "off";

      var classes = panel.classes;
      entry.classes.textContent = classes ? classes.join(" + ") : "All";
      entry.classes.title = classes ? classes.join(", ") : "All classes";
      entry.classes.className = classes ? "filtered" : "";

      var color = panel.box_color || null;
      entry.boxcolor.textContent = color ? titleCase(color) : "Default";
      entry.swatch.style.background = color
        ? (window.Overlay.BOX_COLORS[color] || "transparent")
        : "linear-gradient(90deg,#38bdf8 0 33%,#4ade80 33% 66%,#facc15 66%)";
    });
    syncManualControls();
  }

  // ------------------------------------------------------- command readout

  // Status text shown instead of a result. Auto uses the Korean texts.
  var PLACEHOLDERS = {
    ko: { recognized: "음성 입력을 기다립니다.", command: "명령을 기다립니다.",
          listening: "듣는 중...", recognizing: "인식 중..." },
    en: { recognized: "Waiting for voice input...", command: "Waiting for command...",
          listening: "Listening...", recognizing: "Recognizing..." }
  };
  // null = the readout shows lastRecord; "listening" / "recognizing" / "typed" = a
  // command is being given and no result has arrived yet.
  var readoutPhase = null;
  var readoutText = null;
  var readoutLabel = null;

  function placeholders() {
    return PLACEHOLDERS[currentLanguage() === "en" ? "en" : "ko"];
  }

  function showPlaceholder(node, text) {
    node.textContent = text;
    node.className = "value placeholder";
    node.title = "";
  }

  function renderRecord(record, quiet) {
    lastRecord = record;
    readoutPhase = null;
    var ph = placeholders();
    if (!record || (!record.display && !record.input)) {
      el.inputLabel.textContent = "Recognized";
      showPlaceholder(el.inputText, ph.recognized);
      showPlaceholder(el.commandText, ph.command);
      renderCommandDiag();
      return;
    }
    el.inputLabel.textContent = record.origin === "text" ? "Typed"
      : record.origin === "manual" ? "Manual control" : "Recognized";
    // Recognized is exactly what Whisper returned (or what was typed); it is never
    // corrected here. The parser and Qwen3 work on it only on the server.
    if (record.origin === "manual") {
      el.inputText.textContent = "Controls panel";
      el.inputText.className = "value";
    } else if (record.input) {
      el.inputText.textContent = record.input;
      el.inputText.className = "value";
    } else {
      showPlaceholder(el.inputText, ph.recognized);
    }

    if (!record.display) {
      showPlaceholder(el.commandText, ph.command);
      renderCommandDiag();
      return;
    }
    var display = record.display;
    el.commandText.textContent = "";
    var suffix = " (Qwen3)";
    if (record.qwen_used && display.slice(-suffix.length) === suffix) {
      el.commandText.appendChild(document.createTextNode(display.slice(0, -suffix.length) + " "));
      var tag = document.createElement("span");
      tag.className = "qwen-tag";
      tag.textContent = "(Qwen3)";
      el.commandText.appendChild(tag);
    } else {
      el.commandText.textContent = display;
    }
    el.commandText.className = "value" +
      (record.status === "ok" ? "" : record.status === "error" ? " bad" : " warn");
    el.commandText.title = record.reason || "";
    if (record.qwen_invoked && !quiet) flashChip(el.wl.qwen, 2500);
    renderCommandDiag();
  }

  // Hold to speak is recording.
  function showListening() {
    readoutPhase = "listening";
    var ph = placeholders();
    el.inputLabel.textContent = "Recognized";
    showPlaceholder(el.inputText, ph.listening);
    showPlaceholder(el.commandText, ph.command);
  }

  // A spoken (text null) or typed command was sent and its result is not back yet.
  function showProcessing(text, label) {
    readoutPhase = text ? "typed" : "recognizing";
    readoutText = text;
    readoutLabel = label;
    var ph = placeholders();
    el.inputLabel.textContent = label;
    if (text) {
      el.inputText.textContent = text;
      el.inputText.className = "value";
    } else {
      showPlaceholder(el.inputText, ph.recognizing);
    }
    showPlaceholder(el.commandText, ph.command);
  }

  // Draw the readout again in the current language (after a language switch).
  function refreshReadout() {
    if (readoutPhase === "listening") showListening();
    else if (readoutPhase) showProcessing(readoutText, readoutLabel);
    else renderRecord(lastRecord, true);
  }

  // Clear: display only. The Recognized / Command readout goes back to its
  // placeholder for every viewer; cameras, detection, classes, colours, swap,
  // models and video are not touched (the server keeps them in another object).
  function wireClear() {
    el.clearBtn.addEventListener("click", function () {
      var listening = readoutPhase === "listening";
      renderRecord(null);
      if (listening) showListening();
      postJSON("/api/command/clear", {}).catch(function () {});
    });
  }

  function renderCommandDiag() {
    if (!lastRecord) { el.commandOut.textContent = "no command yet"; return; }
    var r = lastRecord;
    el.commandOut.textContent = [
      "last command",
      "  origin    : " + r.origin + (r.language ? "   language: " + r.language : "") +
        (r.detected_language ? "   detected: " + r.detected_language : ""),
      "  input     : " + (r.input || "—"),
      "  shown     : " + (r.display || "—"),
      "  command   : " + JSON.stringify(r.command),
      "  source    : " + (r.source || "—") + "   qwen invoked: " + !!r.qwen_invoked,
      "  status    : " + r.status + (r.result ? "   (" + r.result.status + ": " + r.result.message + ")" : ""),
      "  reason    : " + (r.reason || "—"),
      "  timing ms : " + JSON.stringify(r.timing_ms || {})
    ].join("\n");
  }

  // ------------------------------------------------------------- workloads

  function flashChip(chip, ms) {
    chip.classList.add("active");
    if (chip === el.wl.qwen) {
      if (qwenFlashTimer) clearTimeout(qwenFlashTimer);
      qwenFlashTimer = setTimeout(function () { chip.classList.remove("active"); }, ms);
    }
  }

  function renderWorkloads(data) {
    var vision = (data && data.vision) || {};
    var det = null, seg = null;
    Object.keys(vision).forEach(function (ch) {
      if (vision[ch].task === "detection") det = vision[ch];
      if (vision[ch].task === "segmentation") seg = vision[ch];
    });
    [[el.wl.det, det, "Object detection"], [el.wl.seg, seg, "Instance segmentation"]].forEach(function (row) {
      var chip = row[0], stats = row[1];
      var fps = stats && typeof stats.yolo_fps === "number" ? stats.yolo_fps : null;
      chip.classList.toggle("live", fps !== null && fps > 0);
      setMeta(chip, fps !== null ? row[2] + " · " + fps.toFixed(1) + " fps" : row[2]);
    });
    var voice = (data && data.voice) || {};
    [[el.wl.whisper, voice.whisper_loaded, "Speech recognition"],
     [el.wl.qwen, voice.qwen_loaded, "Command understanding"]].forEach(function (row) {
      var chip = row[0];
      var ready = !!(voice.ready && row[1]);
      chip.classList.toggle("live", ready);
      chip.classList.toggle("loading", !ready && voice.state && voice.state !== "error");
      setMeta(chip, row[2] + " · " + (ready ? "Ready" : voice.state ? "Loading" : "Offline"));
    });
  }

  function pollWorkloads() {
    fetch("/api/workloads", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(renderWorkloads)
      .catch(function () {});
  }

  // ------------------------------------------------------------------ voice state

  function renderVoice(voice) {
    voiceState = voice;
    var ready = !!(voice && voice.ready);
    var node = el.statusVoice.querySelector("span");
    node.textContent = ready ? "Voice AI ready" : (voice && voice.state ? "Voice AI loading" : "Voice AI offline");
    el.statusVoice.className = "status " + (ready ? "ok" : (voice && voice.state ? "warn" : "bad"));
    if (!recorder.busy && !recorder.context) {
      el.talk.disabled = !ready || !recorderAvailable();
      el.talk.title = recorderAvailable()
        ? (ready ? "Hold to speak (or hold the Space bar)" : "The voice runtime is not ready")
        : "The microphone needs HTTPS and a browser with getUserMedia";
    }
  }

  // ------------------------------------------------------------------- SSE

  function openEvents() {
    var events = new EventSource("/api/events");
    events.onmessage = function (event) {
      var payload;
      try { payload = JSON.parse(event.data); } catch (e) { return; }
      if (payload.state) applyState(payload.state);
      if (payload.command && readoutPhase === "listening") {
        lastRecord = payload.command;          // keep "듣는 중..." while recording
      } else if (payload.command && !(pendingLocal && payload.kind === "command" && payload.command.version === (lastRecord && lastRecord.version))) {
        renderRecord(payload.command);
      }
      if (payload.voice) renderVoice(payload.voice);
      if (payload.kind === "activity") {
        el.wl.whisper.classList.toggle("active", payload.activity === "voice-start");
      }
    };
    events.onerror = function () {
      el.statusVoice.className = "status bad";
      el.statusVoice.querySelector("span").textContent = "UI server lost";
    };
  }

  // ------------------------------------------------------------ text command

  function wireTextCommand() {
    el.textForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var text = el.textInput.value.trim();
      if (!text || el.textRun.disabled) return;
      el.textRun.disabled = true;
      pendingLocal = true;
      showProcessing(text, "Typed");
      postJSON("/api/text/command", { text: text }).then(function (data) {
        if (data.state) applyState(data.state);
        if (data.command) renderRecord(data.command);
        el.textInput.value = "";
      }).catch(function (error) {
        renderRecord({ origin: "text", input: text, display: "Command failed", status: "error",
                       reason: error.message });
      }).then(function () {
        pendingLocal = false;
        el.textRun.disabled = false;
        el.textInput.focus();
      });
    });
  }

  // -------------------------------------------------------- manual controls

  function sendCommand(command) {
    return postJSON("/api/command", { command: command, origin: "manual" })
      .then(function (data) { if (data.state) applyState(data.state); })
      .catch(function (error) {
        renderRecord({ origin: "manual", display: "Control failed", status: "error", reason: error.message });
      });
  }

  function wireManualControls() {
    Array.prototype.forEach.call(el.manual.querySelectorAll(".ctl"), function (group) {
      var position = group.getAttribute("data-position");
      var classSelect = group.querySelector('[data-act="classes"]');
      cfg.canonical_objects.forEach(function (name) {
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        classSelect.appendChild(option);
      });
      var colorSelect = group.querySelector('[data-act="color"]');
      Object.keys(window.Overlay.BOX_COLORS).forEach(function (name) {
        var option = document.createElement("option");
        option.value = name;
        option.textContent = titleCase(name);
        colorSelect.appendChild(option);
      });
      colorSelect.addEventListener("change", function () {
        sendCommand({ action: "set_box_color", camera: position, color: colorSelect.value });
      });
      group.addEventListener("click", function (event) {
        var act = event.target && event.target.getAttribute ? event.target.getAttribute("data-act") : null;
        if (act === "detection-on") sendCommand({ action: "detection_on", camera: position });
        else if (act === "detection-off") sendCommand({ action: "detection_off", camera: position });
        else if (act === "detect-all") sendCommand({ action: "detect_all", camera: position });
        else if (act === "reset") sendCommand({ action: "reset", camera: position });
        else if (act === "apply-classes") {
          var chosen = Array.prototype.filter.call(classSelect.options, function (o) { return o.selected; })
            .map(function (o) { return o.value; });
          if (chosen.length) sendCommand({ action: "detect_only", camera: position, classes: chosen });
          else sendCommand({ action: "detect_all", camera: position });
        }
      });
    });
    el.swap.addEventListener("click", function () { sendCommand({ action: "swap_camera" }); });
  }

  function syncManualControls() {
    if (!state) return;
    Array.prototype.forEach.call(el.manual.querySelectorAll(".ctl"), function (group) {
      var position = group.getAttribute("data-position");
      var panel = state.positions[position];
      var chosen = panel.classes || [];
      Array.prototype.forEach.call(group.querySelector('[data-act="classes"]').options, function (o) {
        o.selected = chosen.indexOf(o.value) >= 0;
      });
      group.querySelector('[data-act="color"]').value = panel.box_color || "auto";
      group.querySelector('[data-role="title"]').textContent =
        titleCase(position) + " · " + (cfg.sources[state.mapping[position]].model || "");
    });
  }

  // ------------------------------------------------------------ diagnostics

  function refreshCameraMessages() {
    fetch("/api/vision/log?limit=40", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var lines = [];
        if (data.vision_log_error) lines.push("NOTE: " + data.vision_log_error);
        if (data.kernel_error) lines.push("NOTE: " + data.kernel_error);
        var links = data.links || {};
        Object.keys(links).sort().forEach(function (key) {
          var e = links[key];
          lines.push("usb " + key + "  " + (e.sensor || "?") + "  -> " + (e.stream || "?"));
          lines.push("   disconnects " + e.disconnects + "  resets " + e.resets +
                     "  URB errors " + e.urb_errors + "  bandwidth refusals " + e.bandwidth_refusals);
          lines.push("   " + e.verdict);
        });
        lines.push("", "counts by severity: " + JSON.stringify(data.counts || {}), "");
        (data.lines || []).forEach(function (entry) {
          lines.push("[" + entry.severity + "] " + entry.line);
        });
        el.cameraOut.textContent = lines.join("\n");
      }).catch(function (error) { el.cameraOut.textContent = "ERROR: " + error.message; });
  }

  function refreshDiagnostics() {
    renderCommandDiag();
    fetch("/api/diagnostics").then(function (r) { return r.json(); }).then(function (data) {
      var lines = [];
      if (cfg.capture) {
        lines.push("capture mode    : " + cfg.capture.mode + " (both cameras, selected by run.sh preflight)");
        Object.keys(cfg.capture.cameras || {}).forEach(function (i) {
          var c = cfg.capture.cameras[i];
          lines.push("  camera " + i + "      : " + (c.device || "?") + "  usb " + (c.usb_path || "?") +
                     "  " + (c.identity || "") + (c.probe_fps ? "  probe " + c.probe_fps + " fps" : ""));
        });
      }
      var channels = data.insight_channels || {};
      Object.keys(sources).forEach(function (key) {
        var src = sources[key];
        var st = src.conn.status();
        var dg = src.conn.diagnostics();
        var ingest = channels[String(src.channel)] || {};
        var local = src.conn.stats();
        lines.push("", "source " + key + " (" + (cfg.sources[key].model || "") + ", channel " + src.channel + ")");
        lines.push("  stream        : " + (st.width || "?") + "x" + (st.height || "?") + " " + (st.codec || "") +
                   "  decoded " + st.fps + " fps  shown " + st.presentedFps + " fps  " + st.kbps + " kbps");
        lines.push("  browser       : jitter buffer " + st.bufferedMs + " ms (target " + st.targetBufferMs +
                   ")  receive->display " + st.browserLatencyMs + " ms  dropped " + st.framesDropped +
                   "  overlay " + (src.drawMs === undefined ? "-" : src.drawMs.toFixed(1)) + " ms");
        lines.push("  insight ingest: active=" + ingest.active + " idr=" + ingest.idr_count +
                   " meta recv=" + ingest.metadata_received + " matched=" + ingest.metadata_matched);
        lines.push("  pairing       : recv=" + local.received + " matched=" + local.matches +
                   " held=" + local.holds + " miss=" + local.misses + " expired=" + local.expired);
        lines.push("  webrtc        : conn=" + dg.connectionState + " peers created " + dg.peersCreated +
                   " offers " + dg.offersSent + " reconnects " + st.reconnects +
                   " media recoveries " + st.mediaRecoveries);
      });
      lines.push("", "ui subscribers: " + data.subscribers + "   uptime: " + data.uptime_s + " s");
      lines.push("", "-- media / command timeline", mediaLog.text(30));
      el.diagOut.textContent = lines.join("\n");
    }).catch(function (error) { el.diagOut.textContent = "ERROR: " + error.message; });
  }

  function setSheet(open) {
    el.sheet.hidden = !open;
    el.scrim.hidden = !open;
    el.controlsToggle.setAttribute("aria-expanded", String(open));
    if (open) {
      refreshDiagnostics();
      refreshCameraMessages();
      diagTimer = setInterval(function () { refreshDiagnostics(); refreshCameraMessages(); }, 2000);
    } else if (diagTimer !== null) {
      clearInterval(diagTimer);
      diagTimer = null;
    }
  }

  function wireSheet() {
    el.controlsToggle.addEventListener("click", function () { setSheet(el.sheet.hidden); });
    el.sheetClose.addEventListener("click", function () { setSheet(false); });
    el.scrim.addEventListener("click", function () { setSheet(false); });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !el.sheet.hidden) setSheet(false);
    });
  }

  // ------------------------------------------------------------------ voice
  //
  // The microphone is the browser's. Audio is captured at 16 kHz mono and sent
  // as a WAV to /api/voice/command with the selected language.

  var WORKLET_SOURCE = [
    "class PcmCollector extends AudioWorkletProcessor {",
    "  process(inputs) {",
    "    const channel = inputs[0] && inputs[0][0];",
    "    if (channel && channel.length) this.port.postMessage(new Float32Array(channel));",
    "    return true;",
    "  }",
    "}",
    "registerProcessor('pcm-collector', PcmCollector);"
  ].join("\n");

  var recorder = { chunks: null, stream: null, node: null, context: null, busy: false };

  function recorderAvailable() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.AudioContext);
  }

  function encodeWav(chunks, sampleRate) {
    var total = 0;
    chunks.forEach(function (chunk) { total += chunk.length; });
    var buffer = new ArrayBuffer(44 + total * 2);
    var view = new DataView(buffer);
    function ascii(offset, text) {
      for (var i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    }
    ascii(0, "RIFF");
    view.setUint32(4, 36 + total * 2, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    ascii(36, "data");
    view.setUint32(40, total * 2, true);
    var offset = 44;
    chunks.forEach(function (chunk) {
      for (var i = 0; i < chunk.length; i += 1) {
        var sample = Math.max(-1, Math.min(1, chunk[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += 2;
      }
    });
    return new Blob([buffer], { type: "audio/wav" });
  }

  function startRecording() {
    if (recorder.busy || recorder.context) return Promise.resolve();
    recorder.chunks = [];
    return navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    }).then(function (stream) {
      recorder.stream = stream;
      var context = new AudioContext({ sampleRate: 16000 });
      recorder.context = context;
      var input = context.createMediaStreamSource(stream);
      var sink = context.createGain();
      sink.gain.value = 0;
      sink.connect(context.destination);
      if (context.audioWorklet) {
        var url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
        return context.audioWorklet.addModule(url).then(function () {
          URL.revokeObjectURL(url);
          var node = new AudioWorkletNode(context, "pcm-collector");
          node.port.onmessage = function (event) { recorder.chunks.push(event.data); };
          input.connect(node);
          node.connect(sink);
          recorder.node = node;
        });
      }
      var processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = function (event) {
        recorder.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      input.connect(processor);
      processor.connect(sink);
      recorder.node = processor;
      return null;
    });
  }

  function stopRecording() {
    var context = recorder.context;
    var chunks = recorder.chunks || [];
    var rate = context ? context.sampleRate : 16000;
    if (recorder.node) {
      try { recorder.node.disconnect(); } catch (e) {}
      if (recorder.node.port) recorder.node.port.onmessage = null;
      recorder.node.onaudioprocess = null;
      recorder.node = null;
    }
    if (recorder.stream) {
      recorder.stream.getTracks().forEach(function (track) { track.stop(); });
      recorder.stream = null;
    }
    if (context) { context.close().catch(function () {}); recorder.context = null; }
    recorder.chunks = null;
    var total = 0;
    chunks.forEach(function (chunk) { total += chunk.length; });
    return { blob: encodeWav(chunks, rate), seconds: total / rate };
  }

  function wireVoice() {
    function idleLabel() {
      el.talkLabel.textContent = "Hold to speak";
      el.talk.classList.remove("recording", "busy");
      el.talk.disabled = !(voiceState && voiceState.ready) || !recorderAvailable();
    }

    function begin(event) {
      if (event) event.preventDefault();
      if (el.talk.disabled || recorder.busy || recorder.context) return;
      startRecording().then(function () {
        el.talk.classList.add("recording");
        el.talkLabel.textContent = "Listening… release to send";
        showListening();
      }).catch(function (error) {
        renderRecord({ origin: "voice", display: "Microphone unavailable", status: "error",
                       reason: error.message });
      });
    }

    function end(event) {
      if (event) event.preventDefault();
      if (!recorder.context || recorder.busy) return;
      var captured = stopRecording();
      el.talk.classList.remove("recording");
      if (captured.seconds < 0.25) {
        idleLabel();
        renderRecord({ origin: "voice", display: "Too short – hold the button while speaking",
                       status: "error" });
        return;
      }
      recorder.busy = true;
      pendingLocal = true;
      el.talk.classList.add("busy");
      el.talkLabel.textContent = "Processing…";
      el.talk.disabled = true;
      showProcessing(null, "Recognized");
      var language = currentLanguage();
      fetch("/api/voice/command?language=" + encodeURIComponent(language), {
        method: "POST", headers: { "Content-Type": "audio/wav" }, body: captured.blob
      }).then(function (response) {
        return response.json();
      }).then(function (data) {
        if (data.state) applyState(data.state);
        if (data.command) renderRecord(data.command);
      }).catch(function (error) {
        renderRecord({ origin: "voice", display: "Voice command failed", status: "error",
                       reason: error.message });
      }).then(function () {
        recorder.busy = false;
        pendingLocal = false;
        idleLabel();
      });
    }

    el.talk.addEventListener("mousedown", begin);
    el.talk.addEventListener("touchstart", begin, { passive: false });
    window.addEventListener("mouseup", end);
    el.talk.addEventListener("touchend", end);
    document.addEventListener("keydown", function (event) {
      if (event.code === "Space" && !event.repeat && event.target === document.body) begin(event);
    });
    document.addEventListener("keyup", function (event) {
      if (event.code === "Space" && event.target === document.body) end(event);
    });
  }

  // ------------------------------------------------------------------- boot

  function fail(message) {
    el.statusCameras.className = "status bad";
    el.statusCameras.querySelector("span").textContent = message;
  }

  function boot() {
    wireFontSize();
    wireSheet();
    if (!window.drawStrategies) {
      fail("Insight renderer missing");
      el.commandText.textContent = "Insight's overlay renderer could not be loaded (/insight/drawing.js).";
      el.commandText.className = "value bad";
      return;
    }
    fetch("/api/config").then(function (r) { return r.json(); }).then(function (data) {
      cfg = data;
      var wanted = Object.keys(cfg.sources);
      if (ONLY_SOURCE !== null && cfg.sources[ONLY_SOURCE]) {
        wanted = [ONLY_SOURCE];
        document.body.classList.add("single-source");
        Object.keys(cfg.sources).forEach(function (k) { if (k !== ONLY_SOURCE) $("cam-" + k).hidden = true; });
      }
      wanted.forEach(buildSource);
      wireManualControls();
      wireLanguage();
      wireTextCommand();
      wireClear();
      wireVoice();
      if (!recorderAvailable()) el.talkLabel.textContent = "Microphone needs HTTPS";
      pollWorkloads();
      setInterval(pollWorkloads, 3000);
      return fetch("/api/state").then(function (r) { return r.json(); });
    }).then(function (snapshot) {
      applyState(snapshot.state);
      renderRecord(snapshot.command);
      renderVoice(snapshot.voice);
      openEvents();
    }).catch(function (error) {
      fail("UI server error");
      el.commandText.textContent = error.message;
      el.commandText.className = "value bad";
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
