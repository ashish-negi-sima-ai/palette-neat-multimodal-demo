# SCOUT: one-camera mission console

SCOUT adds a browser mission console to the MIPI detector: describe an object's
appearance, find it with YOLO26, inspect a real camera crop with Gemma 4 E4B,
and review the image and explanation together. All inference runs on Modalix;
Neat Insight carries the live video and timestamp-matched overlays.

The current DVT stack requires **snapshot inspection**: capture pauses while
Gemma runs, then resumes. Running Gemma while the IMX678 is streaming reproduced
`IPI overflow recovery` followed by a camera watchdog timeout. The same limitation
is documented in this repository's `vision-local/src/mla_gate.h`. Gemma remains
loaded; the camera/YOLO graph restarts after each inspection. This demo
does not claim uninterrupted simultaneous MIPI capture and VLM inference.

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
```

The raw-frame cache retains at most eight samples; only a requested evidence
image is copied into Python. A separate spawned process owns Gemma, with one
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
| `POST /api/reset` | `{}`; cancel the mission and discard its evidence |
| `POST /api/vlm/retry` | `{}`; reload Gemma with capture stopped |
| `GET /api/evidence/<id>.jpg` | Saved image linked by an event |

Generated TLS files and `last-run.json` live under `~/.cache/neat-scout`;
`--runtime-dir` changes this location. `--cert` and `--key` accept existing TLS
credentials. `--summary` optionally writes a second final JSON report.

## Runtime compatibility on this board

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
model judgment, with its evidence available for review. Additional cameras,
cross-camera identity and flight control are not implemented.
