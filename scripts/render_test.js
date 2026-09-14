// render_test.js - drive Insight's real renderer through overlay.js, headless.
//
//     node scripts/render_test.js                  # fetch drawing.js from vf
//     node scripts/render_test.js path/to/drawing.js
//
// This loads the INSTALLED Insight's /static/drawing.js and this project's
// overlay.js into a small fake DOM whose 2D context records every call, then
// draws the exact message shapes the vision pipeline emits and asserts what came out.
//
// It is the one check that cannot be made in Python: that the filter and the
// proven renderer actually compose - that a filtered message really does reach
// drawStrategies, that bounding boxes and polygon masks are really drawn, and
// that "detection OFF" really results in no drawing at all rather than in
// boxes that happen to be off-screen.
//
// What it does NOT prove is that a browser decodes and presents the video.
// Nothing headless can prove that; that is the one step left to a real browser.

"use strict";

const fs = require("fs");
const path = require("path");
const https = require("https");
const vm = require("vm");

const APP_DIR = path.dirname(__dirname);
const failures = [];

function check(label, condition, extra) {
  const mark = condition ? "PASS" : "FAIL";
  console.log(`  [${mark}] ${label}` +
    (condition ? "" : `\n         -> ${JSON.stringify(extra)}`));
  if (!condition) failures.push(label);
}

// --------------------------------------------------------------- fake DOM

function recordingContext(calls) {
  const state = {};
  const handler = {
    get(target, prop) {
      if (prop in target) return target[prop];
      return undefined;
    }
  };
  const ctx = {
    canvas: null,
    save() { calls.push(["save"]); },
    restore() { calls.push(["restore"]); },
    clearRect(...a) { calls.push(["clearRect", ...a]); },
    fillRect(...a) { calls.push(["fillRect", ...a]); },
    strokeRect(...a) { calls.push(["strokeRect", ...a, ctx.strokeStyle]); },
    beginPath() { calls.push(["beginPath"]); },
    closePath() { calls.push(["closePath"]); },
    moveTo(...a) { calls.push(["moveTo", ...a]); },
    lineTo(...a) { calls.push(["lineTo", ...a]); },
    arc(...a) { calls.push(["arc", ...a]); },
    fill() { calls.push(["fill", ctx.fillStyle, ctx.globalAlpha]); },
    stroke() { calls.push(["stroke", ctx.strokeStyle]); },
    fillText(...a) { calls.push(["fillText", ...a]); },
    measureText(text) { return { width: text.length * 7 }; },
    setLineDash(...a) { calls.push(["setLineDash", ...a]); },
    drawImage(...a) { calls.push(["drawImage", a.length]); },
    putImageData(...a) { calls.push(["putImageData"]); },
    getImageData() { return { data: new Uint8ClampedArray(4) }; },
    strokeStyle: null, fillStyle: null, lineWidth: null, font: null,
    globalAlpha: 1, globalCompositeOperation: "source-over",
    imageSmoothingEnabled: true
  };
  return new Proxy(ctx, handler);
}

function makeCanvas(width, height, calls) {
  const canvas = {
    width, height, clientWidth: width, clientHeight: height,
    getContext() { return canvas._ctx; }
  };
  canvas._ctx = recordingContext(calls || []);
  canvas._ctx.canvas = canvas;
  return canvas;
}

function buildSandbox() {
  const offscreenCalls = [];
  const sandbox = {
    console,
    performance: { now: () => 1000 },
    localStorage: { getItem: () => null, setItem: () => undefined },
    ImageData: class { constructor(data, w, h) { this.data = data; this.width = w; this.height = h; } },
    Uint8ClampedArray,
    Number, Math, JSON, Object, Array, String, Boolean, Map, Set, Infinity,
    document: {
      // drawing.js makes one offscreen canvas for RLE masks.
      createElement: () => makeCanvas(64, 64, offscreenCalls)
    }
  };
  sandbox.window = sandbox;
  sandbox.global = sandbox;
  sandbox.offscreenCalls = offscreenCalls;
  return sandbox;
}

// ------------------------------------------------------------ the messages
// Exactly what vision-local/src/stream_worker.cpp emits.

const DETECTION = {
  type: "object-detection", timestamp: 12345, frame_id: "42",
  _insight: { rtp_timestamp: 12345 * 90 },
  data: {
    objects: [
      { id: "0", label: "person", confidence: 0.91, bbox: [10, 10, 40, 80] },
      { id: "1", label: "dog", confidence: 0.72, bbox: [90, 20, 30, 30] },
      { id: "2", label: "chair", confidence: 0.55, bbox: [10, 90, 20, 20] },
      { id: "3", label: "car", confidence: 0.64, bbox: [200, 40, 60, 40] }
    ]
  }
};

const SEGMENTATION = {
  type: "segmentation", timestamp: 999, frame_id: "7",
  _insight: { rtp_timestamp: 999 * 90 },
  data: {
    segments: [
      { id: "0", label: "dog", confidence: 0.80, bbox: [5, 5, 20, 20],
        mask_format: "polygon", mask: [[5, 5], [25, 5], [25, 25], [5, 25]] },
      { id: "1", label: "person", confidence: 0.90, bbox: [40, 5, 20, 40],
        mask_format: "polygon", mask: [[40, 5], [60, 5], [60, 45], [40, 45]] }
    ]
  }
};

// An RLE mask too: the renderer supports it and a future seg model may send it.
const SEGMENTATION_RLE = {
  type: "segmentation", timestamp: 1000, frame_id: "8",
  data: {
    segments: [
      { id: "0", label: "person", confidence: 0.77, bbox: [10, 10, 40, 40],
        mask_format: "rle", mask: { size: [4, 4], counts: [2, 6, 4, 4] } }
    ]
  }
};

const RENDER = {
  video_sync_buffer_ms: 350, metadata_retention_ms: 2000,
  mask_opacity: 0.45, confidence_threshold: 0.0, box_width: 2
};

// ------------------------------------------------------------------- main

function loadDrawingJs(argPath) {
  if (argPath) return Promise.resolve(fs.readFileSync(argPath, "utf8"));
  const config = JSON.parse(
    fs.readFileSync(path.join(APP_DIR, "config", "ui_config.json"), "utf8"));
  // Same resolution as backend/server.py resolve_insight_bases(): an explicit
  // vf_base wins, otherwise it is built from insight.host and insight.vf_port.
  const insight = config.insight;
  let host = (insight.host || "127.0.0.1").trim();
  if (host === "local" || host === "localhost" || host === "") host = "127.0.0.1";
  const base = insight.vf_base ||
    `https://${host.includes(":") ? `[${host}]` : host}:${insight.vf_port || 8081}`;
  const url = new URL(base.replace(/\/$/, "") + "/static/drawing.js");
  return new Promise((resolve, reject) => {
    const request = https.request({
      hostname: url.hostname, port: url.port, path: url.pathname,
      method: "GET", rejectUnauthorized: false, timeout: 8000
    }, (response) => {
      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode} from ${url}`));
        return;
      }
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => { body += chunk; });
      response.on("end", () => resolve(body));
    });
    request.on("error", reject);
    request.on("timeout", () => { request.destroy(new Error("timeout")); });
    request.end();
  });
}

function run(drawingSource) {
  const sandbox = buildSandbox();
  const context = vm.createContext(sandbox);
  vm.runInContext(drawingSource, context, { filename: "insight/drawing.js" });
  vm.runInContext(
    fs.readFileSync(path.join(APP_DIR, "web", "overlay.js"), "utf8"),
    context, { filename: "overlay.js" });

  console.log("\nA  Insight's renderer loaded into a recording canvas -------------");
  check("drawing.js defined window.drawStrategies",
        typeof sandbox.drawStrategies === "object" && sandbox.drawStrategies !== null);
  check("it has the object-detection strategy",
        typeof sandbox.drawStrategies["object-detection"] === "function");
  check("it has the segmentation strategy",
        typeof sandbox.drawStrategies["segmentation"] === "function");
  check("overlay.js loaded on top of it",
        typeof sandbox.Overlay === "object" && typeof sandbox.Overlay.draw === "function");

  const video = { videoWidth: 1280, videoHeight: 720, readyState: 4 };

  function draw(message, plan) {
    const calls = [];
    const canvas = makeCanvas(640, 360, calls);
    const result = sandbox.Overlay.draw(canvas._ctx, canvas, video, 0,
                                        message, plan, RENDER);
    return { calls, result,
             boxes: calls.filter((c) => c[0] === "strokeRect"),
             labels: calls.filter((c) => c[0] === "fillText").map((c) => c[1]),
             fills: calls.filter((c) => c[0] === "fill"),
             polygonPoints: calls.filter((c) => c[0] === "lineTo").length };
  }

  console.log("\nB  bounding boxes, unfiltered -----------------------------------");
  let out = draw(DETECTION, { detection_enabled: true, classes: null });
  check("one box per detection", out.boxes.length === 4, out.boxes.length);
  check("labels carry class and confidence",
        out.labels.some((l) => l.startsWith("person (91%)")), out.labels);
  check("all four classes are labelled",
        ["person", "dog", "chair", "car"].every(
          (name) => out.labels.some((l) => l.startsWith(name + " "))), out.labels);
  check("the canvas is cleared before drawing",
        out.calls[0][0] === "clearRect", out.calls[0]);
  check("boxes are scaled to the canvas, not drawn in frame pixels",
        out.boxes[0][1] === 10 * (640 / 1280) && out.boxes[0][3] === 40 * (640 / 1280),
        out.boxes[0]);
  check("each class gets its own colour",
        new Set(out.boxes.map((b) => b[5])).size === 4,
        out.boxes.map((b) => b[5]));

  console.log("\nC  detect_only: person only -------------------------------------");
  out = draw(DETECTION, { detection_enabled: true, classes: ["person"] });
  check("exactly one box is drawn", out.boxes.length === 1, out.boxes.length);
  check("and it is the person", out.labels.length === 1 &&
        out.labels[0].startsWith("person ("), out.labels);
  check("the UI counts report 1 of 4",
        out.result.kept === 1 && out.result.total === 4, out.result);
  check("the canvas was still cleared, so the old boxes are gone",
        out.calls[0][0] === "clearRect");
  out = draw(DETECTION, { detection_enabled: true, classes: ["bus"] });
  check("a class that is absent draws nothing at all",
        out.boxes.length === 0 && out.labels.length === 0);
  check("and reports 0 of 4", out.result.kept === 0 && out.result.total === 4);

  console.log("\nC2 detect_only: a class list (person + dog) and a forced box colour");
  out = draw(DETECTION, { detection_enabled: true, classes: ["person", "dog"] });
  check("two classes -> two boxes", out.boxes.length === 2, out.boxes.length);
  check("and they are the person and the dog",
        out.labels.some((l) => l.startsWith("person (")) &&
        out.labels.some((l) => l.startsWith("dog (")) && out.labels.length === 2, out.labels);
  check("the UI counts report 2 of 4", out.result.kept === 2 && out.result.total === 4, out.result);
  out = draw(DETECTION, { detection_enabled: true, classes: null, box_color: "green" });
  check("a commanded box colour draws every box in that colour",
        out.boxes.length === 4 && new Set(out.boxes.map((b) => b[5])).size === 1,
        out.boxes.map((b) => b[5]));

  console.log("\nD  detection OFF / ON -------------------------------------------");
  out = draw(DETECTION, { detection_enabled: false, classes: ["person"] });
  check("detection OFF draws nothing", out.boxes.length === 0 &&
        out.labels.length === 0 && out.fills.length === 0);
  check("but it still clears, so nothing is frozen on screen",
        out.calls.length === 1 && out.calls[0][0] === "clearRect", out.calls);
  check("the objects are still counted while OFF",
        out.result.total === 4 && out.result.kept === 0, out.result);
  out = draw(DETECTION, { detection_enabled: true, classes: ["person"] });
  check("detection ON draws the same filter again", out.boxes.length === 1);

  console.log("\nE  segmentation masks ------------------------------------------");
  out = draw(SEGMENTATION, { detection_enabled: true, classes: null });
  check("a box per segment", out.boxes.length === 2, out.boxes.length);
  check("polygon outlines are drawn", out.polygonPoints >= 6, out.polygonPoints);
  check("masks are filled translucently",
        out.fills.length === 2 && out.fills[0][2] === RENDER.mask_opacity,
        out.fills);
  check("segment labels carry class and confidence",
        out.labels.some((l) => l.startsWith("dog (80%)")), out.labels);

  out = draw(SEGMENTATION, { detection_enabled: true, classes: ["person"] });
  check("a segmentation filter keeps mask and box together",
        out.boxes.length === 1 && out.fills.length === 1 &&
        out.labels.length === 1 && out.labels[0].startsWith("person ("),
        { boxes: out.boxes.length, labels: out.labels });
  out = draw(SEGMENTATION, { detection_enabled: false, classes: null });
  check("detection OFF hides masks too",
        out.boxes.length === 0 && out.fills.length === 0);

  out = draw(SEGMENTATION_RLE, { detection_enabled: true, classes: null });
  check("an RLE mask is decoded and blitted",
        out.calls.some((c) => c[0] === "drawImage") &&
        sandbox.offscreenCalls.some((c) => c[0] === "putImageData"),
        out.calls.map((c) => c[0]));

  console.log("\nF  the message is never mutated ---------------------------------");
  const before = JSON.stringify(DETECTION);
  draw(DETECTION, { detection_enabled: true, classes: ["dog"] });
  draw(DETECTION, { detection_enabled: true, classes: null });
  check("filtering twice leaves the arriving message untouched",
        JSON.stringify(DETECTION) === before);
  out = draw(DETECTION, { detection_enabled: true, classes: null });
  check("so an unfiltered draw after a filtered one shows all four again",
        out.boxes.length === 4, out.boxes.length);

  console.log("\nG  no ROI state leaks in from another Insight session -----------");
  // drawing.js reads localStorage['viewerROI_<n>'] unless showRoi is false.
  let roiRead = false;
  sandbox.localStorage.getItem = (key) => {
    if (String(key).startsWith("viewerROI_")) roiRead = true;
    return JSON.stringify([{ type: "inclusion",
                             points: [{ x: 0.9, y: 0.9 }, { x: 0.95, y: 0.9 },
                                      { x: 0.95, y: 0.95 }] }]);
  };
  out = draw(DETECTION, { detection_enabled: true, classes: null });
  check("an ROI polygon in localStorage does not filter our detections",
        out.boxes.length === 4, { boxes: out.boxes.length, roiRead });
  check("and no ROI outline is drawn over the demo",
        !out.calls.some((c) => c[0] === "fill" && c[1] &&
                               String(c[1]).indexOf("0,255,0") >= 0));

  console.log("\nH  video/metadata pairing --------------------------------------");
  vm.runInContext(
    fs.readFileSync(path.join(APP_DIR, "web", "webrtc.js"), "utf8"),
    context, { filename: "webrtc.js" });
  const pair = sandbox.InsightSource._internals;

  // One source's store. A message carries the RTP timestamp of the frame it
  // describes, which is metadata.timestamp * 90 on the wire.
  const store = pair.newStore();
  const stamped = (ms, label) => ({
    type: "object-detection", timestamp: ms, frame_id: String(ms),
    _insight: { rtp_timestamp: ms * 90 },
    data: { objects: [{ id: "0", label, confidence: 0.9, bbox: [0, 0, 1, 1] }] }
  });

  pair.storeMetadata(store, stamped(1000, "person"), 0);
  pair.storeMetadata(store, stamped(1033, "dog"), 10);
  pair.storeMetadata(store, stamped(1066, "car"), 20);

  let hit = pair.takeMetadata(store, 1033 * 90, 0, 30, 0);
  check("a frame takes the message stamped with its own RTP timestamp",
        hit && hit.data.data.objects[0].label === "dog",
        hit && hit.data.timestamp);
  hit = pair.takeMetadata(store, 1000 * 90, 0, 31, 0);
  check("an earlier frame still finds its own message",
        hit && hit.data.data.objects[0].label === "person");
  check("a matched message is consumed, not reused",
        pair.takeMetadata(store, 1000 * 90, 0, 32, 0) === null);

  hit = pair.takeMetadata(store, 9999 * 90, 0, 33, 0);
  check("a frame with no message of its own draws nothing (hold 0)",
        hit === null);
  check("and that is counted as a miss", store.stats.misses === 2,
        store.stats);

  // The hold: only ever a redraw of something that DID match a frame.
  const held = pair.newStore();
  pair.storeMetadata(held, stamped(2000, "person"), 0);
  check("the hold needs a prior match to have anything to hold",
        pair.takeMetadata(held, 7777 * 90, 0, 5, 150) === null);
  pair.takeMetadata(held, 2000 * 90, 0, 10, 150);
  hit = pair.takeMetadata(held, 7777 * 90, 0, 60, 150);
  check("inside the hold window the last match is redrawn",
        hit && hit.data.data.objects[0].label === "person");
  check("outside it, nothing is drawn",
        pair.takeMetadata(held, 7777 * 90, 0, 400, 150) === null);

  // Two sources cannot cross: separate stores, separate connections.
  const storeA = pair.newStore();
  const storeB = pair.newStore();
  pair.storeMetadata(storeA, stamped(3000, "from-source-0"), 0);
  pair.storeMetadata(storeB, stamped(3000, "from-source-1"), 0);
  const fromA = pair.takeMetadata(storeA, 3000 * 90, 0, 1, 0);
  const fromB = pair.takeMetadata(storeB, 3000 * 90, 0, 1, 0);
  check("each source only ever sees its own metadata",
        fromA.data.data.objects[0].label === "from-source-0" &&
        fromB.data.data.objects[0].label === "from-source-1");
  check("and a consumed match in one store leaves the other alone",
        pair.takeMetadata(storeA, 3000 * 90, 0, 2, 0) === null &&
        pair.newStore().timestamped.size === 0);

  // Expiry and the pending cap, which are Insight's own bounds.
  const bounded = pair.newStore();
  for (let i = 0; i < sandbox.InsightSource.MAX_PENDING + 25; i += 1) {
    pair.storeMetadata(bounded, stamped(4000 + i, "person"), i);
  }
  check("pending metadata is capped at Insight's limit",
        bounded.timestamped.size === sandbox.InsightSource.MAX_PENDING,
        bounded.timestamped.size);
  check("the overflow is counted as evicted, not lost silently",
        bounded.stats.evicted === 25, bounded.stats);
  const aging = pair.newStore();
  pair.storeMetadata(aging, stamped(5000, "person"), 0);
  pair.takeMetadata(aging, 6000 * 90, 2000, 5000, 0);
  check("a message older than the retention window expires",
        aging.stats.expired === 1 && aging.timestamped.size === 0,
        aging.stats);

  // The no-timestamp fallback, which is what a browser without
  // requestVideoFrameCallback gets.
  const untimed = pair.newStore();
  pair.storeMetadata(untimed, { type: "object-detection", timestamp: 1,
                                data: { objects: [] } }, 0);
  check("a message with no RTP timestamp still reaches a frame",
        pair.takeMetadata(untimed, undefined, 0, 1, 0) !== null);
  check("and is reported as an arrival fallback, not a match",
        untimed.stats.arrivalFallbacks === 1 && untimed.stats.matches === 0,
        untimed.stats);

  console.log("");
  if (failures.length) {
    console.log(`${failures.length} check(s) FAILED`);
    failures.forEach((label) => console.log(`  - ${label}`));
    process.exitCode = 1;
    return;
  }
  console.log("all checks passed");
}

loadDrawingJs(process.argv[2]).then(run).catch((error) => {
  console.error("could not load Insight's drawing.js: " + error.message);
  console.error("Insight must be running, or pass a path to drawing.js.");
  process.exitCode = 1;
});
