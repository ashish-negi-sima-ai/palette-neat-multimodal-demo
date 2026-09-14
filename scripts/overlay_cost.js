// overlay_cost.js - how much main-thread time the overlay costs per frame.
//
//     node scripts/overlay_cost.js <messages.json> [iterations]
//     node scripts/overlay_cost.js   # uses the shapes render_test.js uses
//
// WHY
//
// The reported fault is "the UI says 19-20 fps and the picture looks like 1".
// Everything upstream of the browser measures ~19-20 fps of genuinely new
// pictures (scripts/fps_forensics.py at the sender, scripts/webrtc_probe.py at
// a real WebRTC peer, both with zero duplicate pictures), so the drop is in
// the browser - and one of the browser's costs belongs to this project: the
// per-frame canvas work in web/overlay.js plus Insight's own drawing.js.
//
// This times exactly that JavaScript, with the INSTALLED Insight renderer and
// real metadata captured off the data channel.  At 20 fps a frame's whole
// budget is 50 ms.
//
// WHAT IT CANNOT MEASURE
//
// Rasterization.  The canvas here is a recording stub, so this is the cost of
// the path building and the geometry, not of the GPU or CPU actually painting
// pixels.  The real figure, including rasterization, is measured in the
// browser itself and shown in each panel's meter as "N ms overlay"
// (web/app.js), which is where the honest number for a given machine is.
// A large number here would be conclusive; a small number here does not
// acquit the browser.

"use strict";

const fs = require("fs");
const path = require("path");
const https = require("https");
const vm = require("vm");

const APP_DIR = path.dirname(__dirname);

// Same render settings the page passes, read from the same config file.
const config = JSON.parse(
  fs.readFileSync(path.join(APP_DIR, "config", "ui_config.json"), "utf8"));
const RENDER = config.render;

function loadDrawingJs() {
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

// A 2D context that accepts everything and records nothing: recording would
// dominate the measurement.
function quietCanvas(width, height) {
  const noop = () => {};
  const ctx = {
    canvas: null,
    save: noop, restore: noop, beginPath: noop, closePath: noop,
    moveTo: noop, lineTo: noop, rect: noop, stroke: noop, fill: noop,
    fillRect: noop, strokeRect: noop, clearRect: noop, fillText: noop,
    strokeText: noop, setLineDash: noop, translate: noop, scale: noop,
    rotate: noop, clip: noop, arc: noop, ellipse: noop, drawImage: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
    createPattern: () => null,
    putImageData: noop,
    createImageData: (w, h) => ({ width: w, height: h,
                                  data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h,
                                     data: new Uint8ClampedArray(w * h * 4) }),
    measureText: (text) => ({ width: String(text).length * 7 }),
    globalAlpha: 1, lineWidth: 1, font: "", fillStyle: "", strokeStyle: "",
    textBaseline: "", textAlign: "", lineJoin: "", lineCap: ""
  };
  const canvas = {
    width: width, height: height,
    clientWidth: width, clientHeight: height,
    style: {}, _ctx: ctx,
    getContext: () => ctx
  };
  ctx.canvas = canvas;
  return canvas;
}

function buildSandbox() {
  const sandbox = {
    console: console,
    performance: { now: () => Number(process.hrtime.bigint()) / 1e6 },
    document: {
      createElement: () => quietCanvas(1920, 1080),
      addEventListener: () => {}, removeEventListener: () => {}
    },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
    addEventListener: () => {}, removeEventListener: () => {},
    setTimeout: setTimeout, clearTimeout: clearTimeout,
    setInterval: setInterval, clearInterval: clearInterval,
    Image: function () {}, ImageData: function () {},
    OffscreenCanvas: function (w, h) { return quietCanvas(w, h); },
    localStorage: { getItem: () => null, setItem: () => {} },
    navigator: { userAgent: "node" },
    JSON: JSON, Math: Math, Date: Date, Object: Object, Array: Array,
    Uint8Array: Uint8Array, Uint8ClampedArray: Uint8ClampedArray,
    Uint32Array: Uint32Array, Float32Array: Float32Array,
    Map: Map, Set: Set, Promise: Promise, isNaN: isNaN, parseInt: parseInt,
    parseFloat: parseFloat, String: String, Number: Number, Boolean: Boolean
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  return sandbox;
}

function percentile(values, fraction) {
  const ordered = [...values].sort((a, b) => a - b);
  const index = Math.min(ordered.length - 1,
                         Math.max(0, Math.round(fraction * (ordered.length - 1))));
  return ordered[index];
}

function main() {
  const messagesPath = process.argv[2];
  const iterations = Number(process.argv[3] || 2000);

  let messages;
  if (messagesPath) {
    messages = JSON.parse(fs.readFileSync(messagesPath, "utf8"));
    if (!Array.isArray(messages)) messages = [messages];
  } else {
    console.error("give a JSON file of captured metadata messages, e.g. the "
                  + "one scripts/webrtc_probe.py --save-metadata writes");
    process.exit(2);
  }

  loadDrawingJs().then((drawingSource) => {
    const sandbox = buildSandbox();
    const context = vm.createContext(sandbox);
    vm.runInContext(drawingSource, context, { filename: "insight/drawing.js" });
    vm.runInContext(
      fs.readFileSync(path.join(APP_DIR, "web", "overlay.js"), "utf8"),
      context, { filename: "overlay.js" });

    const video = { videoWidth: 1920, videoHeight: 1080, readyState: 4 };
    const plan = { detection_enabled: true, classes: null,
                   box_color: null };
    // The panel size that matters: half of a 2560-wide window, 16:9.
    const canvas = quietCanvas(1280, 720);

    console.log("overlay cost: Insight's real drawing.js + web/overlay.js");
    console.log(`  messages   : ${messages.length} from ${messagesPath}`);
    console.log(`  canvas     : ${canvas.width}x${canvas.height}`);
    console.log(`  iterations : ${iterations} per message`);
    console.log("");

    let worstType = null;
    for (const message of messages) {
      const data = message.data || {};
      const list = data.segments || data.objects || [];
      // Warm up, so the first-call compile cost is not reported as the cost.
      for (let i = 0; i < 50; i += 1) {
        sandbox.Overlay.draw(canvas._ctx, canvas, video, 0, message, plan,
                             RENDER);
      }
      const samples = [];
      for (let i = 0; i < iterations; i += 1) {
        const t0 = Number(process.hrtime.bigint());
        sandbox.Overlay.draw(canvas._ctx, canvas, video, 0, message, plan,
                             RENDER);
        samples.push((Number(process.hrtime.bigint()) - t0) / 1e6);
      }
      const median = percentile(samples, 0.5);
      console.log(`  ${message.type.padEnd(16)} ${list.length} `
                  + `entr${list.length === 1 ? "y " : "ies"}: `
                  + `median ${median.toFixed(3)} ms   `
                  + `p95 ${percentile(samples, 0.95).toFixed(3)} ms   `
                  + `max ${percentile(samples, 1).toFixed(3)} ms`);
      if (!worstType || median > worstType.median) {
        worstType = { type: message.type, median: median };
      }
    }

    console.log("");
    const budget = 1000 / 20;
    console.log(`  a frame's whole budget at 20 fps is ${budget.toFixed(0)} `
                + `ms; the worst message above costs `
                + `${worstType.median.toFixed(3)} ms `
                + `(${(100 * worstType.median / budget).toFixed(3)}% of it)`);
    console.log("  This is the JAVASCRIPT only - path building and geometry.");
    console.log("  Rasterization is not included and cannot be measured here;");
    console.log("  the browser's own figure is in each panel's meter as");
    console.log("  \"N ms overlay\", measured per presented frame.");
  }).catch((error) => {
    console.error("could not load Insight's drawing.js: %s", error.message);
    process.exit(1);
  });
}

main();
