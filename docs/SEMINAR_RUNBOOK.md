# Seminar runbook - Palette NEAT SDK Demo

## Before the session

- Connect both C920 webcams to the DevKit's USB hub and leave them on the same ports:
  **port `1-3.1` = LEFT (YOLO26m Detection), port `1-3.3` = RIGHT (YOLO26m Segmentation)**.
  The two C920s have no serial number, so LEFT/RIGHT follow the ports.
- Light the scene. In dim light the webcams cannot hold 30 fps and the preflight selects a
  slower mode (for example 1080p15); the page always shows the mode actually in use.
- Use Chrome on the presentation laptop, full screen (F11) at 1920x1080. The page fits without
  scrolling at the default font size.

## Start

1. `./run.sh --check`; expect 0 FAIL.
2. `./run.sh`. Wait for:
   - `preflight passed`, with USB Camera 0 / USB Camera 1 (device, format, resolution, fps);
   - Camera 0 / Camera 1 READY and Voice backend READY;
   - the `AI workloads on the Modalix MLA` block with all four workloads RUNNING / LOADED.
3. Open `https://<devkit>:8022` and allow the microphone.
4. Check the header: `Cameras 2/2`, `Voice AI ready`, all four workload chips green.

## Presenting

- **Voice.** Hold **Hold to speak** (or hold the Space bar), speak, release. `Recognized` shows
  what Whisper heard, `Command` what was done. `(Qwen3)` after the command means the
  deterministic parser could not decide it and Qwen3-0.6B translated it.
- **Text.** Type into `Type a command…` and press Enter or **Run**. Typed text goes through the
  same command path as recognised speech.
- **Language.** `Auto | KO | EN` next to the microphone button. The page always opens in KO;
  switch at any time.
- **Clear** (top right of Recognized / Command) resets both fields to their waiting text. It only
  clears the display; the cameras, detection settings and LEFT/RIGHT layout stay as they are.
- **Controls** (top right): font size 100 / 125 / 150 %, manual controls per camera (detection
  on/off, classes, box colour, reset, swap) as a backup, and diagnostics.

Good demonstration commands:

| Say or type | Result |
|---|---|
| 왼쪽 카메라에서 사람하고 컵만 탐지해 주세요 | LEFT / PERSON + CUP |
| 오른쪽 탐지 꺼 | RIGHT / DETECTION OFF |
| 왼쪽 바운딩 박스를 그린으로 바꿔주세요 | LEFT / BOX GREEN |
| 왼쪽 카메라 초기화하세요 | LEFT / RESET (detection on, all classes, default colour) |
| 좌우 카메라 바꿔 | SWAP / LEFT ↔ RIGHT |
| Hide the boxes on the right | RIGHT / DETECTION OFF (Qwen3) |
| 오른쪽 카메라 기본 상태로 돌려줘 | RIGHT / RESET (Qwen3) |
| 오늘 날씨 어때 | Command not understood (nothing changes) |

## Emergencies

**Preflight finds only one USB camera**

- Replug the missing camera, check `lsusb` and `vision-local/build-devkit/drone-seminar-usb-vision
  --list-cameras`, then run `./run.sh` again.

**A camera image freezes or disappears**

1. Wait 10 s. The pipeline restarts a silent camera after 5 s, on that camera only.
2. Look at Controls > Camera messages, or `grep -a "\[usb\]\|\[recovery\]" runtime/vision-*.log | tail`.
3. `CAMERA DISCONNECTED` means the camera left the USB bus: replug it (one camera at a time).
4. If the camera is back on the bus but the image does not return within 30 s:
   `./run.sh --stop-all && ./run.sh`.
5. If the kernel log shows `unable to enumerate USB device`, unplug that camera for a few seconds
   and plug it in again. Do not unplug both cameras at the same time.

**Voice shows "Busy – try again"**

- A previous command is still being processed. Wait a second and repeat.

**Voice runtime stops (`Voice AI offline`)**

- `./run.sh --status`, then `./run.sh --stop-all && ./run.sh`. Typed commands keep working with
  the deterministic parser while the voice runtime is down; Qwen3 fallback does not.

**Insight disconnects**

- Restart Neat Insight on the notebook the SDK's usual way (the demo never starts or stops it).
  When `./run.sh --status` shows video on channels [0, 1], reload the page.

**Between long sessions**

- Restart the demo (`./run.sh --stop-all && ./run.sh`): the vision process's memory grows by a few
  MB per voice command inside the NEAT/MLA runtime.
