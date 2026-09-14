// overlay.js - the display filter, and nothing else.
//
// The drawing is NOT done here.  It is done by `window.drawStrategies`, which
// comes from the installed Insight's own /static/drawing.js, served by this app
// at /insight/drawing.js.  So bounding boxes, polygon segmentation masks,
// labels, confidences, colours, letterboxing and canvas scaling are all the
// proven Insight code, unmodified.
//
// This module sits in front of it and does exactly three things:
//
//   detection OFF   draw nothing for this panel; the video keeps playing
//   object filter   hand Insight a message whose list contains only the
//                   matching entries
//   styling         pass Insight a per-label style table through the settings
//                   argument its renderer already accepts, including the
//                   panel's commanded box colour (issue 9)
//
// WHAT IT CANNOT DO
//
// It has no access to the video element beyond passing it to Insight for
// scaling, no access to the RTCPeerConnection and no access to the decoder.
// The strongest thing "detection OFF" can do is skip a clearRect and return
// (spec sections 11, 18).
//
// FILTERING IS BY VALUE
//
// A filtered message is a shallow copy with a new list; the arriving message is
// never mutated.  That matters because the same message can be redrawn by the
// hold path in webrtc.js, and a mutated message would make a filter permanent.

(function (global) {
  "use strict";

  // The two list keys this pipeline emits, by message type
  // (vision-local/src/stream_worker.cpp metadata JSON).
  var LIST_KEY = { "object-detection": "objects", "segmentation": "segments" };

  // A stable colour per class so the same object is the same colour in both
  // panels and across a swap.  Fed to Insight's renderer as its `objects`
  // style table, which is the documented way to colour a label.
  var PALETTE = [
    "#38bdf8", "#4ade80", "#facc15", "#fb923c", "#f472b6",
    "#a78bfa", "#22d3ee", "#fb7185", "#84cc16", "#e879f9"
  ];

  // The commanded colours, by the canonical name backend/panel_state.py's
  // BOX_COLORS uses.  The two tables must agree, so scripts/selftest.py
  // compares them; a name in one and not the other is a failure there, not a
  // silent fallback here.
  var BOX_COLORS = {
    red: "#ef4444",
    green: "#22c55e",
    blue: "#3b82f6",
    yellow: "#facc15",
    white: "#f8fafc",
    orange: "#fb923c",
    purple: "#a855f7",
    cyan: "#22d3ee",
    pink: "#f472b6",
    black: "#0b0f14"
  };

  function colorForLabel(label) {
    var text = String(label || "");
    var hash = 0;
    for (var i = 0; i < text.length; i += 1) {
      hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
    }
    return PALETTE[hash % PALETTE.length];
  }

  function listOf(message) {
    if (!message || typeof message !== "object") return null;
    var key = LIST_KEY[message.type];
    var data = message.data;
    if (!key || !data || typeof data !== "object") return null;
    return Array.isArray(data[key]) ? key : null;
  }

  /**
   * Apply one panel's filter to one metadata message.
   *
   * Returns {message, total, kept} - `message` is null when nothing should be
   * drawn.  `total` and `kept` are what the UI shows as "3 of 11 objects", so
   * the filter is visible rather than asserted.
   */
  function filterMessage(message, plan) {
    if (!message) return { message: null, total: 0, kept: 0 };

    var key = listOf(message);
    if (key === null) {
      // Not a shape this demo knows.  Insight's renderer has strategies for
      // more types than these two, so it is passed through untouched rather
      // than dropped - but it is never filtered, because filtering it would
      // mean guessing its list key.
      return { message: plan.detection_enabled ? message : null,
               total: 0, kept: 0, unknownShape: true };
    }

    var entries = message.data[key];
    var total = entries.length;

    // Spec section 11: detection OFF disables overlay RENDERING.  The message
    // still arrived, the video is untouched, and the count is still reported.
    if (!plan.detection_enabled) {
      return { message: null, total: total, kept: 0 };
    }

    // `classes` is the panel's class list (null = every class).  A single
    // `object_filter` string is still accepted by the offline render tests.
    var wantedList = Array.isArray(plan.classes) && plan.classes.length
      ? plan.classes
      : (plan.object_filter ? [plan.object_filter] : null);
    if (!wantedList) {
      return { message: message, total: total, kept: total };
    }

    var wanted = {};
    for (var w = 0; w < wantedList.length; w += 1) wanted[wantedList[w]] = true;
    var kept = [];
    for (var i = 0; i < entries.length; i += 1) {
      var entry = entries[i];
      if (entry && Object.prototype.hasOwnProperty.call(wanted, entry.label)) kept.push(entry);
    }

    // A shallow copy: the arriving message is left exactly as it came so the
    // redraw path in webrtc.js cannot see a filtered version of it.
    var data = {};
    for (var field in message.data) {
      if (Object.prototype.hasOwnProperty.call(message.data, field)) {
        data[field] = message.data[field];
      }
    }
    data[key] = kept;
    var out = {};
    for (var top in message) {
      if (Object.prototype.hasOwnProperty.call(message, top)) out[top] = message[top];
    }
    out.data = data;

    return { message: out, total: total, kept: kept.length };
  }

  /**
   * The settings object Insight's renderer expects, built from our config.
   *
   * `plan.box_color` is the panel's commanded colour (issue 9).  When it is
   * set, EVERY label in this panel gets that one colour, which is what
   * "왼쪽 카메라 박스 색상을 노란색으로" means.  When it is null the per-class
   * palette is used instead, so the same class is the same colour in both
   * panels and across a swap.  An unknown name falls back to the palette
   * rather than to an invalid stroke - but it cannot arrive, because
   * panel_state.py rejects a colour that is not in its own table.
   */
  function settingsFor(message, plan, render) {
    var labels = {};
    var key = listOf(message);
    if (key) {
      var entries = message.data[key];
      for (var i = 0; i < entries.length; i += 1) {
        var label = entries[i] && entries[i].label;
        if (label) labels[label] = true;
      }
    }
    var forced = (plan && plan.box_color) ? BOX_COLORS[plan.box_color] : null;
    var objects = [{ label: "default", color: forced || "#38bdf8",
                     width: render.box_width || 2, style: "solid" }];
    for (var name in labels) {
      if (Object.prototype.hasOwnProperty.call(labels, name)) {
        objects.push({ label: name, color: forced || colorForLabel(name),
                       width: render.box_width || 2, style: "solid" });
      }
    }
    return {
      // ROI is an Insight viewer feature this demo has no UI for, so it is
      // switched off explicitly rather than left to a localStorage value some
      // other Insight session may have written.
      general: { videoSyncBufferMs: render.video_sync_buffer_ms,
                 metadataRetentionMs: render.metadata_retention_ms,
                 showRoi: false, applyRoiFiltering: false },
      type: { objects: objects,
              confidenceThreshold: render.confidence_threshold || 0,
              maskOpacity: render.mask_opacity }
    };
  }

  /**
   * Draw one frame of one panel.
   *
   * `plan` is {detection_enabled, classes, box_color} for the POSITION this source
   * is currently shown in - so after a swap the same source is drawn with the
   * other position's filter, which is exactly spec section 14.
   */
  function draw(ctx, canvas, video, channel, message, plan, render) {
    // Sizing and clearing happen for every frame regardless of the plan, so
    // "detection OFF" removes the overlay on the very next frame instead of
    // leaving the last boxes frozen on the canvas.
    if (canvas.width !== canvas.clientWidth) canvas.width = canvas.clientWidth;
    if (canvas.height !== canvas.clientHeight) canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var filtered = filterMessage(message, plan);
    if (!filtered.message) return filtered;

    var strategies = global.drawStrategies;
    var strategy = strategies && strategies[filtered.message.type];
    if (!strategy) return filtered;

    strategy(ctx, canvas, filtered.message.data, video, channel, {
      settings: settingsFor(filtered.message, plan, render),
      now: performance.now()
    });
    return filtered;
  }

  global.Overlay = {
    draw: draw,
    filterMessage: filterMessage,
    settingsFor: settingsFor,
    colorForLabel: colorForLabel,
    BOX_COLORS: BOX_COLORS,
    LIST_KEY: LIST_KEY
  };
})(window);
