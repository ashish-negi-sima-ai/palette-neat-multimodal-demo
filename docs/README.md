# Palette NEAT SDK Demo - technical README

## Architecture (everything runs on the Modalix DevKit)

```
USB camera 0 -- v4l2src MJPEG -- neatdecoder NV12 --+--> H.264 --> RTP udp :9000 --+
   (LEFT)                                            +--> YOLO26m (MLA) --> meta :9100 +--> Neat Insight (notebook) --> WebRTC --> browser
USB camera 1 -- v4l2src MJPEG -- neatdecoder NV12 --+--> H.264 --> RTP udp :9001 --+
   (RIGHT)                                           +--> YOLO26m-seg (MLA) -> :9101 -+

browser microphone (16 kHz WAV) --+
browser text input ---------------+--> Demo UI server (backend/server.py, DevKit :8022)
                                                  |
                   voice runtime (voice/board_voice_server.py, DevKit 127.0.0.1:8983, pyneat)
                       Whisper-medium A16W8 (MLA)  -> recognised text --+
                       typed text ----------------------------------------+-> command path
                                                                              (backend/command_processor.py)
                       deterministic parser  -> complete command: execute
                       Qwen3-0.6B A16W4 (MLA) -> only when the parser cannot decide;
                                                strict JSON, validated before execution
```

The vision pipeline, USB camera handling, WebRTC path, Insight integration, camera recovery,
MLA arbitration, preflight and shutdown are the proven `drone_seminar_USB` implementation,
unchanged. The vision binary was not modified (it still builds from `vision-local/` with
`vision-local/build.sh`).

**MLA sharing.** The vision process and the voice runtime share the MLA through a lease on
127.0.0.1:8974 (`VISION_PAUSE_BEHAVIOUR=drop`). While Whisper or Qwen3 runs, YOLO submits no new
inference; the cameras and video keep streaming.

## The four AI workloads

| Workload | Runs on | Runtime | Artifact |
|---|---|---|---|
| YOLO26m Detection (LEFT) | Modalix MLA | NEAT C++ `Model::Runner` | `assets/models/yolo26m.tar.gz` (INT8, in this project) |
| YOLO26m Segmentation (RIGHT) | Modalix MLA | NEAT C++ `Model::Runner` | `assets/models/yolo26m-seg.tar.gz` (INT8, in this project) |
| Whisper-medium | Modalix MLA | pyneat 0.4.0 `genai.ASRModel` | `MODEL_ROOT/whisper-medium-a16w8` (A16W8) |
| Qwen3-0.6B | Modalix MLA | pyneat 0.4.0 `genai.GenAIModel` | `MODEL_ROOT/Qwen3-0.6B-Autoround-a16w4` (A16W4) |

The four workloads are resident together and share the MLA through the arbiter. The YOLO
models infer on every camera frame; Whisper-medium runs once per spoken command and Qwen3-0.6B
only for a command the parser cannot decide. The deterministic parser is ordinary software logic
on the CPU and is not an AI workload.

`./run.sh` ends its startup with a block that confirms each workload on the MLA:

```
  AI workloads on the Modalix MLA:
    YOLO26m Detection      NEAT C++ Model::Runner  -> Modalix MLA       RUNNING (yolo_fps=29.9)
    YOLO26m Segmentation   NEAT C++ Model::Runner  -> Modalix MLA       RUNNING (yolo_fps=29.9)
    Whisper-medium         pyneat genai.ASRModel  -> Modalix MLA        LOADED + WARMED UP
    Qwen3-0.6B             pyneat genai.GenAIModel  -> Modalix MLA      LOADED + WARMED UP
    (the deterministic command parser is ordinary software logic, not an AI workload)
```

## The command path (voice and text are the same)

```
voice:  microphone -> Whisper-medium -> recognised text --+
text:   text input -----------------------------------------+--> command_processor.process(text)
```

`process(text)`, in `backend/command_processor.py`, is the only command interpreter. The voice
runtime calls it for recognised speech (`POST /voice_command`) and for typed text
(`POST /text_command`). If the voice runtime is not reachable, the UI server calls the same
function with no Qwen3, so a typed command still works when the parser can decide it.

1. **Deterministic parser** (`backend/command_normalizer.py`, phrases in
   `backend/command_config.json`). A complete, valid command with every field resolved is executed
   directly. Qwen3 is not used and no `(Qwen3)` marker is shown.
2. **Qwen3-0.6B fallback.** For anything else - unknown, unsupported, incomplete, partial,
   ambiguous - the ORIGINAL text goes to Qwen3-0.6B on the MLA (inside the vision MLA lease).
   The prompt (`voice/command_prompt.txt` plus the few-shot examples in `command_config.json`)
   asks for one JSON object only.
3. **Validation** of the model output before anything is executed:
   - **strict output**: after an empty `<think></think>` template block and whitespace are
     removed, the reply must be exactly one JSON object - no prose, no Markdown, no code fence;
   - **schema**: only the keys `action`, `camera`, `classes`, `color`; a supported action; every
     required field present and valid; no non-empty field the action does not take;
   - **grounding**: the answer may not contain a camera, class or colour the text does not name,
     and may not drop a class or colour the text does name. Camera/display ON-OFF never becomes a
     detection toggle.
4. **Invalid, incomplete, unsupported or ambiguous Qwen3 output** changes nothing - no partial
   command, no guessed field - and the page shows `Command not understood`.
5. The UI server checks the command against the schema again before the panel state applies it.

### The structured command

| action | fields | meaning |
|---|---|---|
| `detect_only` | `camera`, `classes` | draw only these classes (a list of any length); detection ON |
| `detect_all` | `camera` | draw every class again; detection ON |
| `detection_on` / `detection_off` | `camera` | draw / do not draw this camera's overlays |
| `reset` | `camera` | the camera's complete startup state: detection ON, all classes, default box colour |
| `set_box_color` | `camera`, `color` | one box colour: red, green, blue, yellow, white, orange, purple, cyan, pink, black |
| `swap_camera` | - | exchange the LEFT and RIGHT panels |
| `unknown` | `original_text` | nothing is changed |

`camera` is `left`, `right` or `both`. Example: `{"action": "detect_only", "camera": "left",
"classes": ["person", "cup"]}`.

**One camera view is one unit.** A view is its video, its metadata (the same channel), its
detection / segmentation overlay, its model label, its capture status and its overlay state
(detection on/off, classes, box colour). `backend/panel_state.py` stores the overlay state per
source, and `swap_camera` changes only which source is shown on which side, so every part of a
view moves to the other side together. `left` / `right` in a command mean the view shown there
at that moment; the change then stays with that view across later swaps. Swapping twice
restores the original layout and every view's state.

### What the parser understands (examples)

| Command | Result |
|---|---|
| 왼쪽 사람만 보여줘 | LEFT / PERSON ONLY |
| 왼쪽 카메라에서 사람하고 컵만 탐지해 주세요 · 왼쪽에서 person, cup, bottle만 보여줘 | LEFT / PERSON + CUP (+ BOTTLE) |
| 오른쪽 탐지 꺼 · 왼쪽 탐지하지 마 · 왼쪽 카메라 아무것도 탐지하지 마세요 · 오른쪽 detection off | DETECTION OFF |
| 왼쪽 박스를 초록색으로 바꿔줘 · 왼쪽 바운딩 박스를 그린으로 바꿔주세요 · make the left bounding boxes green | LEFT / BOX GREEN |
| 왼쪽 카메라 초기화하세요 · 왼쪽 카메라 리셋하세요 · Reset the left camera | LEFT / RESET |
| 오른쪽 필터 해제 · Show all classes on the right | RIGHT / ALL CLASSES |
| 좌우 카메라 바꿔 · Swap the cameras | SWAP / LEFT ↔ RIGHT |
| 양쪽 탐지 꺼 | BOTH / DETECTION OFF |

The parser never guesses: no camera named means no command; two colours, a class together with
another action, or a conditional sentence (보이면 / if / when) are all left to Qwen3 and its
validation.

**Class lists are never shortened.** In an "only these objects" request (a class is named, or
만 / only is used), every word joined into the list - `X하고`, `X이랑`, `X랑`, `X과`, `X와`,
`X만`, or English `X and Y` / `X, Y` - must be a known class phrase. A word that no deterministic
rule maps to a supported class is kept in the parse result as an *unresolved class term*
(`evidence.unresolved_class_terms`). Qwen3 may still be asked, but no answer is accepted while an
unresolved term exists: it can neither drop the term (`사람 펀만` must not become
`LEFT / PERSON ONLY`) nor guess a class for it. The result is `Command not understood` and nothing
changes. A known deterministic alias does resolve (`사람 컷만` -> `LEFT / PERSON + CUP`), and filler
words (`좀`, `하나`, `이것`, `the`, `all` ...; `matching.list_items` in `command_config.json`) are not
class candidates.

Two Whisper spellings measured on the demo commands are part of the vocabulary: `팜지` for 탐지
and `컷` for 컵 (before 만).

### Qwen3 fallback examples (measured on the DevKit)

| Command (not decided by the parser) | Qwen3 output | Shown |
|---|---|---|
| Hide the boxes on the right | `{"action":"detection_off","camera":"right"}` | RIGHT / DETECTION OFF (Qwen3) |
| Remove the boxes from the left camera | `{"action":"detection_off","camera":"left"}` | LEFT / DETECTION OFF (Qwen3) |
| 오른쪽 카메라 기본 상태로 돌려줘 | `{"action":"reset","camera":"right"}` | RIGHT / RESET (Qwen3) |
| Make the right camera like it was at the start | `{"action":"reset","camera":"right"}` | RIGHT / RESET (Qwen3) |
| 왼쪽 사람이랑 컵만 켜줘 | `{"action":"detect_only","camera":"left","classes":["person","cup"]}` | LEFT / PERSON + CUP (Qwen3) |
| 왼쪽 카메라에서 사람하고 곰돌이만 보여줘 | `{"action":"detect_only","camera":"left","classes":["person","bottle"]}` | Command not understood (bottle is not in the text) |
| 왼쪽 화면 박스 지워줘 | `{"action":"detect_only","camera":"left","classes":["bottle"]}` | Command not understood |
| 오늘 날씨 어때 · 오른쪽 카메라 꺼 | `{"action":"unknown"}` | Command not understood |

pyneat 0.4.0's `GenerationRequest` has no temperature or sampling fields, so the runtime's default
decoding is used. Repeated probes returned identical output for every phrase.

## Language

The page has `Language: Auto | KO | EN` next to the microphone button. Every page load starts in
**KO** (`config/ui_config.json` `voice.language`); the operator can switch at any time, and the
choice lasts for that page. On the DevKit, Auto produced the same Whisper transcript as the fixed
language for all 21 short Korean and English command clips tested.

## The UI

Order on screen: title `Palette NEAT SDK Demo`; `4 AI workloads running concurrently on Modalix
MLA` with the four workloads (live YOLO fps from the vision log, Whisper / Qwen3 state from the
voice runtime; the Whisper chip is highlighted while a spoken command is processed, the Qwen3 chip
when Qwen3 was invoked); the two camera views; `Recognized` (or `Typed`) and `Command`; the
microphone button, language selector and text input; `Controls`.

- **Camera cards.** LEFT / RIGHT label, model name, and the capture mode the running pipeline
  selected (for example `720p30 | H.264`). The mode comes from the preflight selection file
  (`runtime/usb-selection-<host>.env`, passed with `--capture-selection`) and the codec from the
  browser's WebRTC statistics; nothing is hard-coded. No latency figure is shown on the main screen,
  because the browser cannot measure camera-to-screen latency reliably.
- **Recognized / Command.** `Recognized` is the raw Whisper transcript (or the typed text), never
  corrected on screen - for example `왼쪽 카메라에서 사람하고 컷만 탐지해주세요.` - and `Command`
  is what the system understood and executed (`LEFT / PERSON + CUP`, with `(Qwen3)` when Qwen3
  produced it). Without a result both show muted status text: `음성 입력을 기다립니다.` /
  `명령을 기다립니다.` in KO and Auto, `Waiting for voice input...` / `Waiting for command...` in EN.
  While Hold to speak records, `Recognized` shows `듣는 중...` / `Listening...`, and after release
  `인식 중...` / `Recognizing...` until the result arrives; `Command` keeps its waiting text until
  then. One **Clear** button (top right of the section) returns both to the waiting text for every
  viewer. It is display only: cameras, detection, classes, box colours, LEFT/RIGHT mapping, models
  and video are not changed.
- **Text.** Every text area has its own box. Recognised and command text reserve two lines, wrap
  (Korean keeps words whole) and scroll inside their own box when longer, so nothing overlaps.
- **Controls** (slide-over panel): font size 100 / 125 / 150 %, manual controls per camera
  (detection on/off, all classes, reset, class list, box colour) and swap, Diagnostics (per stream
  decoded / shown fps, bitrate, jitter buffer, receive-to-display time, pairing, the last command
  with its reason and timing) and Camera messages.
- **Screen fit.** At 100 % the page is exactly one viewport high at 1920x1080 (and at 1920x960 with
  browser chrome, and 1366x768); at 125 % it still fits 1920x1080; at 150 % it scrolls by about
  20 px.

## USB cameras

**Discovery** (`vision-local/src/usb_camera.cpp`) never uses a fixed `/dev/videoN`. A node is a
candidate only if its sysfs device is a USB interface, its own V4L2 `device_caps` report
VIDEO_CAPTURE and STREAMING (this rejects the UVC metadata node), and it offers MJPG or YUYV modes.

**Identity.** `usb-serial:<vid:pid>:<serial>` when the camera has a unique serial number,
otherwise `usb-path:<bus-port.port>@<vid:pid>`. Two identical C920s without serial numbers are
therefore told apart by their USB port. **Order** is by USB topology, so camera 0 (LEFT) and
camera 1 (RIGHT) stay the same while the cameras stay on their ports.

Reference DevKit (two Logitech C920, no serial number, one USB 2.0 hub):

| Port | Identity | Camera | Panel |
|---|---|---|---|
| `1-3.1` | `usb-path:1-3.1@046d:08e5` | camera 0 | LEFT, YOLO26m Detection |
| `1-3.3` | `usb-path:1-3.3@046d:08e5` | camera 1 | RIGHT, YOLO26m Segmentation |

**Capture mode.** `USB_CAMERA_MODES` is a preference list. The preflight streams each candidate on
both cameras at once and takes the first one both deliver at 75 % or more of its frame rate. Two
C920s on one USB 2.0 bus are refused 1080p30/24/20 (`VIDIOC_STREAMON: No space left on device`)
and get MJPG 1280x720 @ 30 (29.9 fps measured on both). In dim light the cameras cannot hold 30 fps
and a slower mode (for example 1080p15) is selected; the page shows whichever mode is in use.

**Recovery** is unchanged: a silent or erroring camera is rebuilt after 5 s on that camera only; a
disconnected camera is looked for every 2 s and picked up again under a new `/dev/videoN`.

## Commands

| Command | What |
|---|---|
| `./run.sh [--background]` | preflight; vision, voice runtime, UI; MLA workload confirmation |
| `./run.sh --status` | processes, cameras, Insight channels, voice readiness |
| `./run.sh --stop-all` | stop UI, voice and vision (only this copy's processes, verified by pid file and /proc) |
| `./run.sh --check` | `scripts/check_environment.sh` (read-only) |
| `./run.sh --selftest` | `scripts/selftest.py` (offline) |

**Ctrl+C** in the `run.sh` terminal stops, in order, the UI server, the voice runtime and the
vision process (SIGINT; TERM/KILL after 40 s), and removes the pid and selection files and the
SDK's `/tmp/sima_gst_plugin_scanner_<pid>`.

## Configuration

- `config/default.env` (+ `config/local.env`): `MODEL_ROOT`, addresses, ports, USB camera selectors
  and modes, `PYNEAT_ENV`.
- `config/ui_config.json`: UI port/TLS, Insight, default language, render settings.
- `backend/command_config.json`: classes, aliases, intents, actions, Qwen3 few-shot examples.
- `voice/command_prompt.txt`: the Qwen3 system prompt.
- `config/vision.yaml`: inference thresholds, recovery timings, MLA lease.

## Logs (all under `runtime/`)

| File | Content |
|---|---|
| `vision-<host>.log` | camera selection, per-camera `[stats]` every 5 s, MLA lease windows, USB recovery |
| `voice-<host>.log` | model load, warm-up, every Whisper / Qwen3 inference and the command it became |
| `ui-<host>.log` | UI server requests and every applied command |
| `preflight-<host>.log` | USB camera discovery and mode probe |
| `usb-selection-<host>.env` | the selected cameras and mode (removed on stop) |

The SiMa SDK also writes fixed paths outside the project: `/tmp/sima_gst_plugin_scanner_<pid>`
(removed by `run.sh`), `/tmp/sima_gst_registry_*`, `/tmp/rpmsg_lock_*`,
`/tmp/neat_boxdecode_segment_contract.txt` and the model unpack cache under `/media/nvme/simaai`.

## Test tools

| Tool | What |
|---|---|
| `scripts/selftest.py` | offline: architecture, parser, command path, panel state, renderer, page, Node tests |
| `backend/command_normalizer.py --selftest`, `command_processor.py --selftest`, `panel_state.py --selftest` | unit self-tests (scripted Qwen3 answers) |
| `scripts/command_acceptance.py` | end to end against the running UI: text and voice, reset, Qwen3, not-understood, multi-class, language, equivalence |
| `scripts/qwen_probe.py` | Qwen3-0.6B on the MLA with the real prompt and validator (demo stopped) |
| `scripts/make_tts_clips.py` | the voice test clips in `testdata/commands/` (TTS) |
| `scripts/portability_audit.py` | symlinks and references to other workspace projects |

## Measured on the reference DevKit (2026-09-14)

Hardware: two Logitech C920 (no serial number) on USB ports 1-3.1 / 1-3.3, MJPG 1280x720 @ 30.
Voice was tested with TTS clips (`testdata/commands/`) uploaded through `POST /api/voice/command`,
the endpoint the page's microphone button uses; a live microphone in a room was not tested.

**Functional run** (full `./run.sh`):

| Check | Result |
|---|---|
| Preflight | passed; 1080p30/24/20 refused for USB bandwidth, 720p30 selected, 29.9 fps on both cameras |
| YOLO | 29.9 fps on both cameras |
| Voice runtime warm-up | Whisper-medium load + warm-up 9.4 s; Qwen3 structured self-test `{"action":"detect_only","camera":"left","classes":["person"]}` 452 ms |
| Browser (headless Chromium, real WebRTC) | both streams 1280x720 H.264, 26-30 fps shown, mode label `720p30 | H.264` |
| Command acceptance (text + voice) | 108 / 112; the 4 failures are two stress clips Whisper mishears (below) |
| Voice / text equivalence | 27 / 27 recognised texts, typed, gave the identical command and display |
| Ctrl+C | `run.sh` exit 130 after 17.7 s; no process, port, pid file, `/dev/video*` holder or scanner file left |

**Latency** (ms, voice runtime timings and the caller's wall clock):

| Stage | median | p90 | max |
|---|---|---|---|
| Whisper-medium on the MLA | 488 | 642 | 805 |
| spoken command, parser, voice runtime total | 501 | 655 | 689 |
| spoken command, parser, end to end | 660 | 903 | 972 |
| spoken command with Qwen3, end to end | 911 | 998 | 1080 |
| Qwen3-0.6B on the MLA, per command | 206 | 296 | 311 |
| typed command, parser, end to end | 24 | 25 | 27 |
| typed command with Qwen3, end to end | 242 | 321 | 335 |

**Stability run** (full `./run.sh`, 08:23-08:55, 31.8 min unattended). Load during the run: one
real WebRTC viewer on both streams (headless Chromium), a command every 45 s alternating a
spoken clip and a typed command (40 commands, Qwen3 fallback on 12), health samples every 15 s
(`runtime/test/run2_stability/`).

| Measure | Camera 0 (LEFT, detection) | Camera 1 (RIGHT, segmentation) |
|---|---|---|
| RTP to Insight | 1,094,044 packets, 551-594 pps per 15 s interval, 0 empty intervals | 1,098,829 packets, 479-597 pps, 0 empty intervals |
| Metadata | 29.7 msg/s | 29.7 msg/s |
| YOLO fps (5 s windows) | median 29.9, min 26.8 | median 29.9, min 26.7 |
| Browser viewer | 30 fps decoded, 24-31 shown (median 29.5), 0 reconnects | 30 fps decoded, 25-31 shown (median 28), 0 reconnects |
| Mode label | `720p30 | H.264` throughout | `720p30 | H.264` throughout |

| System-wide | Result |
|---|---|
| Commands | 40 / 40 correct (20 voice, 20 text); voice end to end median 685 ms (max 1167), Whisper median 502 ms |
| Camera restarts / runner restarts / USB disconnects, resets, bandwidth, URB errors / OOM | 0 / 0 / 0 / 0 |
| Processes | same PIDs throughout; vision RSS 227 -> 229 MB, 101 threads constant, CPU median 250 %; voice RSS 201 MB flat; UI 28 -> 33 MB |
| Board | MemAvailable min 4254 MB, CmaFree min 844 MB, load1 max 5.2, max 55.0 C |
| Page | 1080 px high throughout (no scrolling), 0 console errors |
| Ctrl+C at the end | `run.sh` exit 130 after 17.9 s; no process, port, pid file, `/dev/video*` holder or scanner file left |

**Cold start** (software reboot of the DevKit - a physical power cycle was not possible remotely -
then `/workspace/drone_seminar_neat/run.sh`, `runtime/test/run3_coldstart/`):

| Check | Result |
|---|---|
| Boot | booted 08:56:26; `/workspace` mounted, both C920s enumerated on ports 3.1 / 3.3 |
| `run.sh` | started 08:57:16; preflight passed; ready at 08:58:19 (63 s) |
| Cameras | LEFT `usb-path:1-3.1`, RIGHT `usb-path:1-3.3` - the same mapping as runs 1 and 2 |
| Capture mode | MJPG 1280x720 @ 30, 29.9 fps on both |
| AI workloads | YOLO26m Detection / Segmentation 29.9 fps; Whisper-medium and Qwen3-0.6B loaded and warmed up, all on the Modalix MLA |
| Insight | connected, video on channels 0 and 1 |
| Demo UI | title, MLA statement, 4 live workload chips, `720p30 | H.264` on both cards, default language Auto; fits 1920x1080 at 100 % and 125 % |
| Voice and text commands | 43 / 43 (required, reset, Qwen3 fallback, not understood, multi-class; by voice and by text) |
| Ctrl+C | exit 130 after 17.7 s; no process, port, pid file, `/dev/video*` holder or scanner file left |

## Known limitations

- **Voice tested with TTS.** All voice results are from TTS clips, not a person at a microphone in a
  room.
- **Whisper mishearings.** Measured: a slow Korean reading of "오른쪽 탐지 꺼" became "오른쪽 팜지과"
  and an Australian-English "Right detection off" became "Write detection off". Both were safely not
  understood (nothing changed); the speaker has to repeat.
- **Qwen3-0.6B is small.** It answers `unknown` for some paraphrases (for example 왼쪽 박스 없애줘,
  오른쪽 박스 좀 숨겨줘) and sometimes invents a class; the validator refuses those, so the result is
  `Command not understood`, never a wrong action.
- **No sampling control.** pyneat 0.4.0 exposes no temperature / greedy setting for Qwen3.
- **Resolution.** Two USB 2.0 cameras on one bus run at 720p30, not 1080p30.
- **Light.** In dim light the capture mode drops (for example 1080p15).
- **Vision memory after voice commands.** The vision process's RSS grows by a few MB per
  Whisper/Qwen3 command inside the NEAT/MLA runtime (inherited, see the stability result). Restart
  the demo between long sessions.
- **Camera identity follows ports.** Two identical serial-less cameras swapped between ports swap
  LEFT and RIGHT.
