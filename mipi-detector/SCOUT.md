# SCOUT: MIPI + USB mission console

SCOUT adds a browser mission console to the MIPI detector: describe an object's
appearance, find it with YOLO26, inspect a real camera crop with Gemma 4 E4B,
and review the image and explanation together. All inference runs on Modalix;
Neat Insight carries the live video and timestamp-matched overlays.

An optional USB view adds landing/payload area monitoring. Switch between
**Find a subject** on MIPI and **Watch an area** on USB; both live views remain
visible, with camera-labelled observations in one evidence timeline.

The [three-camera architecture and visitor use case](ARCHITECTURE.md) describes
the planned addition of an e-con IMX568 global-shutter camera with integrated
FPGA alongside the existing IMX678 and eMeet Nova.

SCOUT uses **snapshot inspection with continuous capture**: it copies a camera
crop for Gemma while the MIPI and USB camera/YOLO runs and H.264 encoders remain
live. Enabled speech models stay resident too. Finishing or cancelling a request
does not unload a model or deinitialize a camera.

All SCOUT models belong to one lifetime group. Stopping or losing a model stops
the entire group, including both camera/YOLO runs and encoders. **Restart all
models** closes that group and fully exits Python before the launcher starts a
new process, releasing process-owned native resources and allocator caches.
This replaces the older per-inspection
pause/reload workaround; those earlier measurements are historical below.

## Start on the SoM devkit

The Waveshare SoM devkit at `10.42.0.147` uses camera `imx678 6-0042`
with `modalix-som-waveshare-ECON-IMX678-1CAM.dtbo`. A direct 1080p NV12
capture delivered 30 FPS. Start SCOUT with that camera and frame rate:

```bash
ssh sima@10.42.0.147
cd /workspace/GitHub/palette-neat-multimodal-demo
./mipi-detector/scout.sh --camera 'imx678 6-0042' --fps 30 \
  --host 10.42.0.1 --allow-cpu-fallback
```

Open **<https://10.42.0.147:8022>**. The model directory defaults to
`/media/nvme/llima/models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy`. Allow initial model loading
to finish before the camera starts. The DVT-specific `SCOUT_LIBRARY_PATH`
override below is separate from this SoM launch command.

The installed SoM camera stack requires `--allow-cpu-fallback`: strict capture
failed with “Required zero-copy DMA-BUF pool is unavailable”. With the fallback,
Neat copies camera buffers into its device-memory pipeline; YOLO preprocessing
and inference still run on the accelerators. A 60-frame detector smoke test
completed successfully with this option.

Gemma and enabled speech services initialize before camera capture begins.
After startup, inference does not rebuild camera or detector graphs. H.264
encoders stay open until the whole model group stops; video and detection
metadata continue sharing the same source timestamps during inspection.

## Add the USB landing/payload view

The connected eMeet Nova provides MJPEG at 1280×720. Start both cameras with:

```bash
./mipi-detector/scout.sh --camera 'imx678 6-0042' --fps 30 \
  --host 10.42.0.1 --allow-cpu-fallback --usb-camera auto
```

`auto` selects the sole `/dev/v4l/by-id/*-video-index0` camera. For multiple
USB cameras, pass that camera's full stable path with `--usb-camera`.
On the current board it resolves to `/dev/video97`; numbering can change.
Omit `--usb-camera` for the original MIPI-only application.

1. Point USB at a table or mat and select **Watch an area**.
2. Click **Draw watch area**, then drag over the USB image to mark the region.
3. Place a cup, bottle, backpack, or another watched object in the region.
   Stable occupancy saves an **Area occupied** image; removing watched objects
   produces **No watched objects** after a one-second settling interval.
4. Click **Inspect area snapshot** to ask Gemma about the exact 480×480 crop.
   The answer and image join the shared timeline. Both cameras keep streaming
   while Gemma inspects that saved image.
5. Switch to **Find a subject** to use MIPI appearance checks. Its reset and
   new missions preserve USB evidence.

The default watch classes are person, backpack, handbag, suitcase, bottle, cup,
bowl, laptop, cell phone, book, and chair. The table surface itself is excluded.
You can select one class instead. An object is in the area when at least 20%
of its detection box overlaps it; occupancy settles for 0.5 seconds, and absence
for one second. Paused, disconnected, and stale views report unknown occupancy.
These are detector observations, not a landing clearance or a guarantee that
every obstruction is visible or recognized.

USB processing defaults to 15 FPS (`--usb-fps`), with `--usb-width 1280` and
`--usb-height 720`. OpenCV captures and decodes USB MJPEG; YOLO runs through the
public Neat Model API. Video uses a persistent Neat H.264 encoder and Insight
channel `--channel + 1`. Metadata and video share each frame's receive timestamp;
USB evidence labels it as received PTS rather than a sensor hardware timestamp.
Unplugging USB makes that view unavailable and triggers reconnect attempts;
MIPI does not depend on USB availability. Stop SCOUT with Ctrl+C or SIGTERM.

Both detector runs and encoders remain alive during Gemma and speech requests.
Track IDs are local (`T…` for MIPI, `U…` for USB) and continue through inspection. Cross-camera identity/handoff remains a future step.

## Optional voice input and spoken replies

The [voice architecture](VOICE_ARCHITECTURE.md) adds browser audio while retaining
one MIPI IMX678 plus USB. Configuration is in
[scout-voice.example.yaml](scout-voice.example.yaml). On the currently connected
board, the IMX678 is named `imx678 6-0042`, as set in that YAML.

```bash
ssh sima@10.42.0.147
cd /workspace/GitHub/palette-neat-multimodal-demo
./mipi-detector/scout.sh --config mipi-detector/scout-voice.example.yaml
```

Open <https://10.42.0.147:8022>. This enables Whisper Medium and Supertonic.
Add `--no-stt` to disable recognition, or `--no-tts` to disable spoken replies.
Both `speech.stt.enabled` and `speech.tts.enabled` are independent YAML switches;
`speech.enabled: false` disables both. CLI settings override YAML. Launches without
`--config` preserve the original app with speech disabled.

With STT enabled, choose the subject class, press **Listen**, speak its appearance,
then press **Stop & inspect**. SCOUT stops the browser microphone, transcribes on
Modalix, and submits the transcript as a mission with inspection requested. It
waits for a stable YOLO candidate before taking the existing Gemma snapshot.
The 30-second recording limit stops the mic but requires **Submit recording**;
silence never submits a mission. Invalid/overlong text remains available to edit.
Reset, editing the mission, changing its class, switching views, or Cancel speech
invalidates pending voice work. No microphone is opened automatically.

Supertonic reads accepted subject and USB snapshot results, once per evidence
item, on the initiating browser. It does not read occupancy events or raw JSON.
Playback stops when listening begins. If autoplay is blocked, use **Play reply**.
The browser device needs HTTPS microphone permission; Modalix needs no microphone
or speaker. The existing self-signed certificate must be accepted first.

SCOUT reuses `/workspace/GitHub/sima-ai/services/speech/start.py` and its
`jarvic.audio.clients` through the configured Jarvic Python environment. Its own
HTTPS server proxies audio; the model servers bind only to `127.0.0.1`. Enabled services
start once with SCOUT and remain running between requests. An occupied service port is
reported rather than taking over another app's server.

Dependencies: PyYAML in SCOUT's Python, the installed `sima-ai/jarvic_venv` with
its Jarvic dependencies, and `onnxruntime==1.22.1` in the configured Supertonic
Python (normally `/home/sima/pyneat/bin/python3`). Supertonic's ONNX text encoder
and duration predictor use the CPU; its vector field and vocoder use the MLA.
All model paths point under `/media/nvme/llima/models` in the example YAML.

The camera/YOLO runs, Gemma and enabled speech models are resident together;
Gemma and speech inference can run alongside live detection. The supplied YAML
requires `serialize_inference: false`, `pause_capture_for: []`,
`speech_worker_lifetime: application`, `release_gemma_for_speech: false`, and
`stop_all_models_together: true`. The old pause/on-demand settings are rejected.
Changes to model enable flags require restarting the application.

Request cancellation discards late results without stopping a model. Native
model-process failure or a hard request deadline stops the whole group. Every
owner receives a stop signal; cleanup continues through other owners even if
one close fails. A failed group cleanup prevents automatic restart. External
model services are never taken over or stopped by SCOUT.

**Restart all models** (`POST /api/vlm/retry`, retained for compatibility) stops
cameras, YOLO, Gemma, speech servers and encoders, releases graph/model references,
then exits Python with status 75. `scout.sh` waits for that process to exit before
starting a new Python process. Launch through `scout.sh` to support this action;
running `scout.py` directly exits without restarting. Missions and in-memory
evidence reset on this full restart. An unexpected model exit shuts the application down;
start it again with the launch command above.

WAV recordings are deleted after processing. One reply's audio is retained until
replacement, cancellation or shutdown. Written evidence remains available if an
individual speech request fails without terminating a model. Speech-server logs
are under `speech.runtime_dir/models-*/`; those contain no browser recordings.

Validated on 2026-09-28 with Whisper disabled: two consecutive USB snapshot
inspections produced Gemma verdicts and spoken Supertonic replies while both
cameras stayed live. Browser video frames and matched overlays advanced during
Gemma and TTS; no camera stop/restart occurred and the model PIDs did not change.
Steady-state MIPI / USB rates were 30 / 15 FPS (throughput may vary under load).
Stopping the owned Supertonic server stopped the complete group in 2.62 seconds,
with zero cleanup errors. Reported DMA allocations returned from 96 objects /
200,376,320 bytes to the baseline of 0 objects / 0 bytes. Test artifacts are under
`runtime/scout-resident-20260928/`.

The browser restart action fully exited the old Python process, started a new
model group and reconnected both live views with matched overlays. Stopping the
launcher with SIGTERM stopped all owners in 2.48 seconds; native allocations
returned to the system-only baseline (8 buffers, about 3 MiB) and DMA buffers
returned to zero. All 38 unit tests passed both locally and on Modalix.

Whisper was subsequently enabled and validated on the same board. A known speech
recording transcribed exactly as “A person wearing a dark shirt.” in 1.394 seconds
through the sima-ai client. The browser Listen / Stop path took 4.128 seconds from
Stop to automatic mission submission, including WAV processing, request-worker
startup and transcription. Gemma inspected the live MIPI person in 1.829 seconds
and Supertonic played the accepted result; a USB check also completed in 1.625
seconds with a spoken reply. Both cameras continued streaming with matched
overlays, and all model PIDs stayed unchanged between requests.

The browser test used Chromium's file-backed microphone with actual AudioWorklet
recording, resampling, upload and on-board inference. It verified no microphone
access on page load, microphone release on Stop / Cancel, and no upload after
Cancel. This validates the software path, not the user's physical microphone or
an acoustic listening assessment. Artifacts: `runtime/scout-whisper-20260928/`.

Stopping Whisper shut down the complete model group in 3.066 seconds, with no
cleanup errors. DMA buffers returned to zero and native allocations returned
to the system-only baseline (8 buffers, about 3 MiB). SCOUT was then started
again with both speech features enabled.

The configuration and speech ownership tests load no models. Run them with:

```bash
python3 -m unittest discover -s mipi-detector -p 'test_scout*.py' -v
```

## Historical DVT setup

Stop the plain MIPI detector or the USB/voice demo before starting SCOUT; they
share the accelerator, camera/viewer resources, or console port.

```bash
ssh sima@10.42.0.73
cd /workspace/GitHub/palette-neat-multimodal-demo

# This board's installed Neat needs its matching LLiMa runtime; see below.
export SCOUT_LIBRARY_PATH="$HOME/.cache/neat-scout/runtime-compat/usr/lib/aarch64-linux-gnu"

./mipi-detector/scout.sh --host 10.42.0.1 \
  --vlm-model /workspace/llima/models/gemma-4-E4B-it-GPTQ-a16w4
```

Open **<https://10.42.0.73:8022>**. The first visit may require accepting the
local certificate. Allow the model to finish loading before inspecting a subject.
Ctrl+C stops the worker, graph and console. SIGTERM and SIGHUP also shut down cleanly.

The launcher uses `~/pyneat/bin/python`; override `PYTHON` or `PYNEAT_ENV` when
needed. Dependencies are the installed pyneat, NumPy, OpenCV and `openssl`.
The web server uses Python's standard library. Model files remain external.

## Visitor walkthrough

1. Put a clearly lit person, backpack, cup or another supported object in view.
2. Choose its class and describe a visible detail, such as “a blue backpack”.
3. Click **Inspect subject**. SCOUT waits for a stable detection and saves its
   crop for Gemma. Both live views and detection continue during the check.
4. Read **Snapshot matches**, **Snapshot does not match**, or **Need a clearer
   view**. Click the evidence thumbnail to see the exact 480×480 image Gemma saw.
5. Subject Focus appears only after a matching snapshot. SCOUT checks again every
   three seconds while the mission is active. Reset stops subject checks.

Subject Focus is a live digital crop of the same track whose snapshot Gemma
confirmed. Candidates are boxed in the overview but do not appear in focus
before a positive verdict. A negative, uncertain or failed recheck clears that
track's focus. Lost detections show no crop; once the track expires (0.8 seconds),
confirmation is discarded. A newly appearing ID always needs its own match.
This is a digital crop, not a gimbal command. The optional
watch zone reports the selected **object class**, independently of its appearance
verdict. It uses the bottom-center of each box in image coordinates.

With multiple detections, SCOUT tries stable tracks of the selected class that
have not been checked, then the least recently checked. YOLO confidence breaks
ties. Once a subject matches, subsequent checks prioritize that identity. A
rejection returns to searching; it does not repeatedly prefer the same
high-confidence nonmatch over unchecked subjects.

Automatic checks are enabled by default. `scout.inspection_interval: 3.0` in YAML
or `--inspection-interval 3` sets the minimum interval between snapshot starts.
The first check starts as soon as a stable candidate is available. There is one
shared Gemma request at a time: slower inference or USB inspection delays the
next subject check, without building a queue of old snapshots. Cameras remain
live throughout. Recording/transcribing a new voice mission suspends checks of
the previous description. Repeated same-track verdicts remain in the evidence
timeline; speech repeats only when the track or verdict changes. A new submitted
mission resets this speech suppression. Use `--no-auto-checks` (or YAML
`auto_checks: false`) for checks only on explicit requests.

Validated on Modalix on 2026-09-28: repeated negative results left Subject Focus
empty, a positive result enabled only its matching track, voice recording
suspended the old mission's checks, and Reset stopped checks. Recorded snapshot
intervals were approximately 3.0 seconds with both camera views and matched
overlays advancing. Candidate rotation and clearing confirmation on negative,
uncertain, failed and stale results are covered by the state tests. Artifacts:
`runtime/scout-focus-20260928/`.

```bash
./mipi-detector/scout.sh --mission 'a blue backpack' --object-class backpack
./mipi-detector/scout.sh --inspection-interval 3 --mission 'a person wearing an orange vest'
./mipi-detector/scout.sh --no-auto-checks --mission 'a blue backpack' --object-class backpack
./mipi-detector/scout.sh --help
```

## Data flow and interfaces

```text
MIPI NV12 ──┬── H.264 → Neat Insight → browser video
            ├── YOLO26 → class-aware tracking → matched browser overlays
            └── bounded frame cache → matching detection/camera PTS
                                      ↓ on inspection
                       crop + letterbox + JPEG (480×480)
                                      ↓ copied evidence; capture and YOLO stay live
                           Gemma → JSON verdict + reason
                                      ↓
                       saved evidence + optional spoken reply

USB MJPEG → OpenCV BGR ──┬── H.264 → Neat Insight channel + 1
                        ├── YOLO26 → area occupancy → saved event images
                        └── latest image → area crop → same Gemma worker
```

The raw-frame cache retains at most eight samples and forwards shared camera
tensors to the persistent encoder; only a requested evidence image is copied
into Python. A separate spawned process owns Gemma, with one
outstanding request and a timeout. Malformed output cannot become a positive
verdict. Resetting or changing a mission invalidates its in-flight answer.
Evidence is bounded to the latest 24 events and lives in memory.

The console proxies WebRTC signalling to Insight at `--host` and
`--insight-vf-port` (8081 by default). Video and metadata use the original
detector's `--channel`, `--video-port-base` and `--metadata-port-base` settings.
If port 8022 is occupied, select another console `--port`.

| Interface | Purpose |
|---|---|
| `GET /api/state` | Mission, capture, VLM, timing and bounded event history |
| `GET /api/config` | Camera dimensions, channel and supported classes |
| `POST /api/mission` | `{ "query": "a blue backpack", "object_class": "backpack", "zone": false }` |
| `POST /api/reset` | `{}`; cancel the MIPI mission and discard its evidence; preserve USB events |
| `POST /api/vlm/retry` | `{}`; stop all models/cameras/encoders, then restart the application |
| `POST /api/watch` | Update USB `{ "enabled": true, "object_class": "any", "zone": [x,y,w,h] }`; normalized coordinates |
| `POST /api/watch/inspect` | `{}`; inspect a recent USB area snapshot when Gemma is ready |
| `GET /api/evidence/<id>.jpg` | Saved image linked by an event |

Generated TLS files and `last-run.json` live under `~/.cache/neat-scout`;
`--runtime-dir` changes this location. `--cert` and `--key` accept existing TLS
credentials. `--summary` optionally writes a second final JSON report.

## Runtime compatibility on the previous DVT

The installed Neat build is `0.4.0+feature-gemma4-mtp.f9a195bbf049`. The newer
system LLiMa runtime answered a request but aborted during model destruction.
The matching package already on the board was extracted privately, with no
system package replacement:

```bash
mkdir -p "$HOME/.cache/neat-scout/runtime-compat"
dpkg-deb -x \
  "$HOME/sima-lmm-0.4.0+feature-gemma4-mtp.409715343926-Linux-core.deb" \
  "$HOME/.cache/neat-scout/runtime-compat"
```

`SCOUT_LIBRARY_PATH` selects those libraries for this application only. A direct
Gemma image request and model destruction succeeded with that matching runtime.
On another board use a compatible Neat/LLiMa release set rather than copying
this board-specific package choice blindly.

## Checks

```bash
python3 -m unittest discover -s mipi-detector -p 'test_scout*.py' -v
node --check mipi-detector/scout-web/scout.js
```

All 45 unit tests pass, including camera timestamp continuity, evidence
matching, USB occupancy, generation ownership, resident speech service lifetime,
complete cleanup after a model-owner failure, and full process exit before a
requested restart, confirmed-track focus, candidate rotation and the three-second
inspection cadence. Current board results are in
the voice section above.

### Historical pause/resume implementation

The following measurements predate the resident model-group change.

The two-camera SoM validation used the e-con IMX678 and eMeet Nova together:
MIPI detection/streaming at 1920×1080/30 FPS and USB at 1280×720/15 FPS.
Browser tests exercised camera switching, drawing an area, the saved-image
dialog, and Gemma checks on both cameras. The USB area check took 1.49 seconds
of VLM time; the MIPI chair check took 1.66 seconds. Camera stop/restart and
browser recovery add to these times. Both streams resumed with exact RTP
metadata matches and no arrival-time fallbacks. There were no JavaScript errors
or horizontal overflow at desktop and 390-pixel mobile widths.

An eight-second USB driver disconnect left MIPI running at 30 FPS. USB reported
offline with unknown occupancy, then automatically recovered capture, video and
matched overlays after the driver was rebound. The current room view was used
throughout; no staged landing mat or cross-camera handoff was tested.

The earlier MIPI-only validation on the Waveshare SoM devkit delivered 1080p
browser video and detection at about 30 FPS, two successive Gemma snapshot
checks (2.52 and 1.45 seconds of VLM time), and resumed capture with exact RTP
metadata matches and no arrival-time fallbacks. Both evidence dialogs opened
the saved 480×480 crop; no JavaScript errors occurred.

Validated on the DVT with e-con IMX678 at CSI-2 CONN_0: 1,500 live frames at
25 FPS, browser video at 1920×1080/25 FPS, exact RTP metadata matching, and an
actual Gemma snapshot check followed by resumed capture. The first backpack
check took about 1.8 seconds of VLM time and correctly returned uncertainty for
a dark crop. Camera stop/restart adds to that time. The original plain detector
also passed a separate finite smoke test.

Browser tests also exercised positive and negative answers, evidence dialogs,
repeated inspections, reset during generation and malformed HTTP input. Both
completed inspections resumed video and matched overlays; no JavaScript errors
occurred. Desktop and 390-pixel mobile layouts were checked for overflow.

Good lighting and lens focus matter: a detection box does not mean a crop has
enough detail to judge clothing or object color. Gemma's verdict remains a
model judgment, with its evidence available for review. A second MIPI camera,
cross-camera identity and flight control are not implemented.
