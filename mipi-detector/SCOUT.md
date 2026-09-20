# SCOUT: MIPI + USB mission console

SCOUT adds a browser mission console to the MIPI detector: describe an object's
appearance, find it with YOLO26, inspect a real camera crop with Gemma 4 E4B,
and review the image and explanation together. All inference runs on Modalix;
Neat Insight carries the live video and timestamp-matched overlays.

An optional USB view adds landing/payload area monitoring. Switch between
**Find a subject** on MIPI and **Watch an area** on USB; both live views remain
visible, with camera-labelled observations in one evidence timeline.

SCOUT uses **snapshot inspection**: capture pauses while
Gemma runs, then resumes. Running Gemma while the IMX678 is streaming reproduced
`IPI overflow recovery` followed by a camera watchdog timeout on the DVT. The same limitation
is documented in this repository's `vision-local/src/mla_gate.h`. Gemma remains
loaded; the camera/YOLO graph restarts after each inspection. This demo
does not claim uninterrupted simultaneous MIPI capture and VLM inference.

## Start on the SoM devkit

The Waveshare SoM devkit at `10.42.0.147` uses camera `imx678 5-0042`
with `modalix-som-waveshare-ECON-IMX678-1CAM.dtbo`. A direct 1080p NV12
capture delivered 30 FPS. Start SCOUT with that camera and frame rate:

```bash
ssh sima@10.42.0.147
cd /workspace/GitHub/palette-neat-multimodal-demo
./mipi-detector/scout.sh --camera 'imx678 5-0042' --fps 30 \
  --host 10.42.0.1 --allow-cpu-fallback
```

Open **<https://10.42.0.147:8022>**. The model directory defaults to
`/workspace/llima/models/gemma-4-E4B-it-GPTQ-a16w4`. Allow initial model loading
to finish before the camera starts. The DVT-specific `SCOUT_LIBRARY_PATH`
override below is separate from this SoM launch command.

The installed SoM camera stack requires `--allow-cpu-fallback`: strict capture
failed with “Required zero-copy DMA-BUF pool is unavailable”. With the fallback,
Neat copies camera buffers into its device-memory pipeline; YOLO preprocessing
and inference still run on the accelerators. A 60-frame detector smoke test
completed successfully with this option.

The application retains snapshot inspection on the SoM: it pauses capture
while Gemma checks a crop, then resumes. The H.264 encoder stays alive in a
separate run: restarting it alongside YOLO on the installed Neat 0.4.0 runtime
caused `infra.accelerator_execution_failed` in preprocessing. Camera and YOLO
can restart while that encoder remains open. Video and detection metadata use
the same timestamp offset to preserve alignment across camera restarts.
Concurrent VLM/camera operation on the SoM has not been validated.

## Add the USB landing/payload view

The connected eMeet Nova provides MJPEG at 1280×720. Start both cameras with:

```bash
./mipi-detector/scout.sh --camera 'imx678 5-0042' --fps 30 \
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
   The answer and image join the shared timeline. Both cameras pause during
   the check, then resume automatically.
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

Both detector runs drain and close before Gemma executes. Both encoders stay
alive through the pause. Track IDs are local (`T…` for MIPI, `U…` for USB) and
renewed across pauses. Cross-camera identity/handoff remains a future step.

## Start on the validated DVT

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
3. Click **Inspect subject**. SCOUT waits for a stable detection, saves its crop,
   and shows a capture-pause banner while Gemma checks it.
4. Read **Snapshot matches**, **Snapshot does not match**, or **Need a clearer
   view**. Click the evidence thumbnail to see the exact 480×480 image Gemma saw.
5. Live detection resumes automatically. Inspect again for another observation,
   or reset the mission. Repeated checks of the same description retain evidence.

The focus panel follows a live YOLO candidate of the requested class. It is a
digital crop, not a gimbal command. Track IDs are discarded across capture pauses;
a snapshot verdict never marks a newly appearing track as verified. The optional
watch zone reports the selected **object class**, independently of its appearance
verdict. It uses the bottom-center of each box in image coordinates.

For an unattended exhibit, `--auto-checks` repeats inspections with a 15-second
cooldown after each answer. The default checks only when requested.

```bash
./mipi-detector/scout.sh --mission 'a blue backpack' --object-class backpack
./mipi-detector/scout.sh --auto-checks --mission 'a person wearing an orange vest'
./mipi-detector/scout.sh --help
```

## Data flow and interfaces

```text
MIPI NV12 ──┬── H.264 → Neat Insight → browser video
            ├── YOLO26 → class-aware tracking → matched browser overlays
            └── bounded frame cache → matching detection/camera PTS
                                      ↓ on inspection
                       crop + letterbox + JPEG (480×480)
                                      ↓ capture stopped; YOLO drained
                           Gemma → JSON verdict + reason
                                      ↓
                       saved evidence; resume live capture

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
| `POST /api/vlm/retry` | `{}`; reload Gemma with capture stopped |
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
python3 -m unittest discover -s mipi-detector -p test_scout.py -v
node --check mipi-detector/scout-web/scout.js
```

All 22 local unit tests pass, including camera timestamp continuity, evidence
matching, USB occupancy settling, invalid/stale areas, independent track IDs,
and preserving USB evidence across MIPI mission changes.

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
