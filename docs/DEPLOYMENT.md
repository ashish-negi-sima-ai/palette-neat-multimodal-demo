# Deploying the Palette NEAT SDK Demo to another Modalix DevKit

You need only this project directory and two GenAI model directories. No MIPI camera, no
device-tree camera overlay and no other demo project is required.

## 1. Copy

**Project directory.** Copy the whole `drone_seminar_neat/` directory to the DevKit, for example
to `/workspace/drone_seminar_neat`. Any location works: every path is resolved from the script
location. `runtime/` may be left out; it is recreated.

The copy must include:

- `vision-local/build-devkit/drone-seminar-usb-vision` (the vision binary)
- `assets/models/yolo26m.tar.gz` and `assets/models/yolo26m-seg.tar.gz` (the YOLO archives)

**GenAI models.** Copy these two directories under the model root (`MODEL_ROOT`, default
`/media/nvme/llima/models`):

- `whisper-medium-a16w8/`
- `Qwen3-0.6B-Autoround-a16w4/`

For example, from a machine that has them:

```
rsync -a --exclude runtime/ drone_seminar_neat/ sima@<devkit>:/workspace/drone_seminar_neat/
rsync -a /media/nvme/llima/models/whisper-medium-a16w8 \
         /media/nvme/llima/models/Qwen3-0.6B-Autoround-a16w4 \
         sima@<devkit>:/media/nvme/llima/models/
```

`python3 scripts/check_models.py --machine devkit --sha256` checks all four models and verifies the
SHA256 of both YOLO archives. `python3 scripts/portability_audit.py` checks that the copy has no
symlink and no path into any other project.

## 2. Connect two USB UVC cameras

Plug in two USB webcams (the seminar setup is two Logitech C920). `/dev/videoN` numbers do not
matter.

- **Identity.** A camera with a unique USB serial number is identified by it. Two identical
  cameras without a serial number (C920s often report none) are identified by their **USB port
  path**, so they are still told apart.
- **Order.** Cameras are ordered by USB port path: the lower path is camera 0 = LEFT = YOLO26m
  Detection. On the reference DevKit, hub port `1-3.1` is LEFT and `1-3.3` is RIGHT.
- **Keep the ports.** If two identical cameras are moved to other ports, LEFT and RIGHT follow the
  ports, not the cameras.

```
vision-local/build-devkit/drone-seminar-usb-vision --list-cameras
```

lists the attached cameras, their USB identity and every capture mode.

## 3. Configure (only what differs from the defaults)

```
cp config/example.env config/local.env      # then edit
```

- `MODEL_ROOT`: where the two GenAI model directories are.
- `DEVKIT_IP`, `HOST_IP`, `INSIGHT_HOST`: the DevKit, and the notebook that runs Neat Insight.
- `PYNEAT_ENV`: the pyneat environment (default `$HOME/pyneat`).
- `USB_CAMERA0_DEVICE`, `USB_CAMERA1_DEVICE`: optional fixed camera selection, for example
  `path:1-3.1` / `path:1-3.3` or `serial:D91EAE8F`.
- `USB_CAMERA_FILTER`: only needed with more than two cameras attached.
- `USB_CAMERA_MODES` / `USB_CAMERA_MODE`: capture-mode preference list, or one forced mode.
- `config/ui_config.json` `voice.language`: the page's default recognition language
  (`auto`, `ko` or `en`).

## 4. If the vision binary must be rebuilt

Only needed for a different SDK version. In the SiMa SDK container:

```
vision-local/build.sh
```

## 5. Check and run

```
cd /workspace/drone_seminar_neat
./run.sh --check        # optional, read-only environment check
./run.sh
```

`run.sh` runs its preflight first; nothing starts if a camera, model, library, port or stale
instance problem is found. It prints the selected capture mode for both cameras, starts vision,
the voice runtime and the UI, and ends with a block that lists the four AI workloads and confirms
each one is on the Modalix MLA. Open `https://<devkit>:8022` in Chrome. Ctrl+C stops everything
that run started.

## Required on the target DevKit (normal SDK / BSP / OS)

- Modalix BSP / SDK 2.1.3 with NEAT (`libsima_neat`) and the SiMa GStreamer plugins
  (`neatdecoder`, `neatencoder`), plus GStreamer's `v4l2src`
- the `mlashmcomplex` MLA service
- pyneat 0.4.0 with genai support in `PYNEAT_ENV`
- python3, curl, openssl
- the `uvcvideo` kernel driver
- Neat Insight on the notebook (`INSIGHT_HOST`), and a Chromium-based browser with a microphone
