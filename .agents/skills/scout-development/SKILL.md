---
name: scout-development
description: "Extend or debug SCOUT and the standalone MIPI detector in palette-neat-multimodal-demo. Use for camera pipelines, mission tracking, Gemma snapshot checks, evidence storage, USB watch regions, or the SCOUT browser UI under mipi-detector/."
---

# SCOUT development

Work from this repository's current implementation. Read
[SCOUT.md](../../../mipi-detector/SCOUT.md) for behavior and runtime constraints,
then inspect the files relevant to the request. Paths below are relative to the
checkout root; do not depend on a particular `/workspace` mount.

## Find the implementation

| Change | Starting point |
|---|---|
| Camera, preprocessing, YOLO configuration and decoded boxes | `mipi-detector/main.py`: `make_model`, `make_graph`, `detection_objects` |
| SCOUT launch, HTTP API, shared VLM scheduling and capture lifecycle | `mipi-detector/scout.py`: `run`, `Console`, `Preview`, `FrameCache` |
| Resident speech servers and coordinated model shutdown | `mipi-detector/scout_speech.py`, `scout_speech_worker.py`, `scout_lifecycle.py` |
| MIPI tracking, missions and shared evidence | `mipi-detector/scout_state.py`: `Tracker`, `MissionState` |
| USB capture, area occupancy, reconnect and area inspection | `mipi-detector/scout_watch.py`: `WatchState`, `UsbWatch` |
| Gemma artifacts, prompts, worker process and cancellation | `mipi-detector/scout_vlm.py` |
| Layout, controls, warning colors and overlays | `mipi-detector/scout-web/{index.html,scout.css,scout.js}` |
| Shared WebRTC transport and frame/metadata correlation | `web/webrtc.js` |
| Hardware-independent behavioral checks | `mipi-detector/test_scout.py` |

SCOUT starts through `mipi-detector/scout.sh`; the plain detector uses
`mipi-detector/run.sh`. The root `run.sh` launches the separate USB/voice demo
described in [docs/README.md](../../../docs/README.md). Changes to shared
`web/webrtc.js` can affect both applications; prefer SCOUT's own renderer for
SCOUT-only presentation changes.

## Preserve the relevant contracts

- **Model-group lifetime:** Gemma and enabled speech models load once before
  capture starts. Both camera/YOLO runs and encoders stay live during snapshot
  and speech inference. The user requires every SCOUT model to stop together
  when one model stops. Fatal model loss stops the group; explicit restart
  closes all owners and fully exits Python before `scout.sh` starts a new
  process, releasing native descriptors and allocator caches. Do not replace
  Python in place or restore the historical per-request pause/reload
  workaround. Request cancellation invalidates results without unloading models.
- **Image and time ownership:** MIPI uses NV12; USB uses OpenCV BGR with matching
  Neat preprocessing. `FrameCache` keeps at most eight samples. Evidence lookup
  uses original MIPI capture PTS; preview video and metadata use the same
  restart offset. USB video, detections and evidence share frame receive PTS.
  Avoid mixing wall-clock timestamps with stream PTS.
- **Snapshot identity:** Keep the image, camera, mission/watch generation,
  source timestamp and MIPI track ID attached to each request. Reject answers
  from an outdated generation. Gemma consumes the same decoded 480×480 JPEG
  displayed as evidence. A snapshot verdict does not verify a new live track.
  Keep one active VLM request, bounded pending work, cancellation and timeout.
- **Confirmed subject focus:** Only a positive snapshot may select its still-live
  track for Subject Focus. A negative/uncertain/error recheck, new mission or
  expired identity clears it. Never fall back to an unverified detection in the
  focus renderer. Automatic subject checks default to three seconds between
  snapshot starts, with no overlapping Gemma requests or catch-up queue. Try
  unchecked/least-recently-checked candidates until a match, then recheck that
  identity. Repeated same-track verdicts retain evidence without repeating speech.
- **Independent missions:** MIPI IDs use `T…`, USB IDs use `U…`; IDs are local
  and tracks continue through inspection. MIPI mission changes preserve
  USB evidence. Scope cancellation to the relevant camera's request.
- **USB observations:** `WatchState` validates normalized `[x,y,w,h]` regions.
  Current occupancy requires at least 20% detection-box overlap, with 0.5-second
  presence and one-second absence settling. Paused, offline and stale capture
  mean unknown occupancy. The default payload classes exclude the table itself.
  Keep a detector observation distinct from Gemma's snapshot verdict.
- **Evidence lifetime:** `MissionState` holds JPEG bytes and the latest 24
  events in memory. `/api/evidence/<id>.jpg` serves those bytes. Shutdown writes
  metadata to `last-run.json`, without persisting JPEGs. If asked for durable
  storage, implement that deliberately rather than treating the existing URLs
  or summary file as an archive.
- **Presentation:** Draw detections from timestamp-matched metadata and reject
  outdated watch/mission generations. The USB region pulses dark red when
  occupied, with a steady tint for reduced-motion users. Clear warnings for
  unavailable observations. These are visual monitoring missions; cross-camera
  identity association and flight control are not implemented.

The user's requested behavior can change these contracts. Update the owning
state, UI, validation and documentation together when it does.

## Implement and verify

Use installed public `pyneat` APIs. For an API change, inspect the matching
installed headers/bindings and the nearest available Neat Apps example before
coding. If available, `neat-application-builder` provides that workflow; these
repository instructions do not require a personal skills installation.

For state or lifecycle changes, run from the repository root:

```bash
python3 -m unittest discover -s mipi-detector -p 'test_scout*.py' -v
```

For browser changes, run `node --check mipi-detector/scout-web/scout.js` and
inspect the changed behavior in a browser. A small color/layout change needs
visual verification rather than new tests that merely repeat its implementation.
Check desktop/mobile layout or reduced motion when affected. Static assets are
read on request, so a browser refresh usually suffices for UI-only edits.

Hardware-independent tests do not establish camera or accelerator correctness.
For pipeline changes, validate on the requested board: check each configured
camera, inspect a snapshot, and confirm advancing video with matched overlays
during inference. For lifetime changes, verify group shutdown/restart and that
owned processes and DMA buffers are released. Respect any user request to defer
testing a model (for example Whisper while it downloads).
Use [scout-modalix-operations](../scout-modalix-operations/SKILL.md) for this work.
Report what was actually run and update `SCOUT.md` for changed behavior. Keep
model archives in `models/` and generated `runtime/` artifacts out of code commits
unless the user explicitly requests them.
