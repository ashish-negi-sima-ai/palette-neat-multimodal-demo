# MIPI YOLO26 object detector

A standalone Python application combining the Neat
[MIPI capture example](../../sima-neat/apps/examples/benchmarking/mipi-camera-capture/README.md)
and [YOLO26 detector example](../../sima-neat/apps/examples/object-detection/yolo26-object-detector/README.md).
It uses this repository's `models/yolo26m-det-int8-b1.tar.gz` and COCO labels.

For a mission console with natural-language appearance checks, Gemma 4 and
clickable snapshot evidence, see [SCOUT](SCOUT.md).

```text
MIPI camera (NV12) --+--> H.264 encoder --> RTP/UDP --> Neat Insight
                    +--> YOLO26 on MLA --> boxes --> metadata --> browser overlay
```

Capture uses the MIPI example's 32 application-owned DMA-BUFs. Neat handles
NV12-to-RGB conversion, 640×640 letterboxing, normalization, inference, and
YOLO26 decoding. Video and detections retain the same camera timestamps.
Only bounding-box data is copied into Python; camera images stay in the graph.

## Run on the DVT board

Prerequisites: a working MIPI camera (`cam -l`), the installed `~/pyneat`
environment, the supplied compiled model archive, and Neat Insight on the host.
No Python package installation or compilation is needed on the validated board.

For the e-con IMX678 connected to DVT `CSI-2 CONN_0`, the boot configuration is
`dtbos=modalix-dvt-ECON-IMX678-0CAM.dtbo`. The camera is named `imx678 0-0042`.

```bash
ssh sima@10.42.0.73
cd /workspace/GitHub/palette-neat-multimodal-demo
./mipi-detector/run.sh --host 10.42.0.1
```

Open the Insight viewer on the host:

<https://10.42.0.1:8081/static/viewer.html?mode=light&src=0>

The default output is 1920×1080 at a requested 25 FPS, H.264 on UDP 9000 and
object-detection metadata on UDP 9100 (channel 0). The console prints FPS,
labels, and scores once per second. Ctrl+C stops the app and releases the camera,
encoder, and model. SIGTERM and SIGHUP also request a clean shutdown.

Use a free Insight channel when another application is streaming. If your SDK
publishes different host ports, pass `--video-port-base` and `--metadata-port-base`
using its `neat --json` port map, and use `/api/viewer-url` for the viewer URL.
The two-camera voice demo and this app should be run separately: this simple
detector does not participate in that demo's MLA voice lease.

## Options

```bash
# Explicit camera and model, with a lower detection threshold.
./mipi-detector/run.sh --camera 'imx678 0-0042' \
  --model models/yolo26m-det-int8-b1.tar.gz --score 0.25 --host 10.42.0.1

# A finite smoke test; no background process remains after completion.
./mipi-detector/run.sh --frames 100 --summary /tmp/mipi-detector-summary.json

# Inference and console output without video or metadata transmission.
./mipi-detector/run.sh --no-stream --frames 100

./mipi-detector/run.sh --help
```

Default model and label paths are resolved relative to this repository, so the
launcher works from any directory. Explicit relative paths use the working
directory. Set `PYNEAT_ENV` or `PYTHON` to use a different installed environment.

The default camera path requires zero-copy capture. `--allow-cpu-fallback`
explicitly permits Neat's compatibility copy path. A missing model or label file,
an invalid option, a camera/build error, or no inference output for
`--timeout-ms` causes a nonzero exit. `--print-backend` prints graph diagnostics.

### Frame-rate limit on this board

`--fps` requests a capture rate; the camera driver can clamp it. The installed
IMX678/libcamera stack reports `mode rate limit: 25 fps max (40000 us/frame)`.
A 1080p streaming test requesting 60 FPS still delivered about 25 detections
per second after startup (400 frames, 23.7 FPS including startup and shutdown).
Requesting 1280×720 at 60 FPS failed camera configuration: the ISP adjusted the
sensor format to 1920×1080. Keep the validated 1920×1080 at 25 FPS settings.
Higher live FPS requires a faster supported sensor mode in the camera stack
before application or model optimizations can help.

## Troubleshooting

- No camera: check `cam -l` and the DVT overlay before running the app.
- No image in Insight: confirm `--host` is the notebook's address, not the
  DevKit's address; check `/api/ingest/stats` on Insight port 9900.
- Video but no boxes: focus the lens, put a COCO object such as a person or cup in view, and
  check the console's object count. Check that metadata arrives on the same
  channel as video, and open the port-8081 viewer with a hard refresh.
- Camera already in use: stop the other camera application first.
- Model allocation errors: stop other MLA inference applications first.

`models/` contains the separately supplied model artifact; it is not copied into
this application directory. The existing USB/voice demo is launched by the root
`run.sh`; this detector is launched by `mipi-detector/run.sh`.

## Validation

Tested on the Modalix DVT board with the e-con IMX678 on `CSI-2 CONN_0`,
the supplied YOLO26m archive, and the board's installed pyneat 0.4.0 build.
A 1,500-frame streaming run completed at 24.6 inference FPS with no metadata
send failures. The browser decoded 1920×1080 video at 25 FPS and displayed
person detection boxes. A 50-frame console-only run also detected people.
Finite runs and SIGTERM released the camera, allowing the next run to start.
