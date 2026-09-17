#!/usr/bin/env python3
"""MIPI camera -> YOLO26 -> Neat Insight video and object-detection overlays."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None, configure=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo26m-det-int8-b1.tar.gz")
    parser.add_argument("--labels", type=Path, default=ROOT / "assets/coco_labels.txt")
    parser.add_argument("--camera", default="", help="Name from cam -l; empty selects the default")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--capture-buffers", type=int, default=32)
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Permit copying if camera DMA-BUF negotiation fails")
    parser.add_argument("--score", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--host", default="10.42.0.1", help="Host running Neat Insight")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--video-port-base", type=int, default=9000)
    parser.add_argument("--metadata-port-base", type=int, default=9100)
    parser.add_argument("--bitrate", type=int, default=4000, help="H.264 bitrate in kbps")
    parser.add_argument("--no-stream", action="store_true", help="Run detection with console output only")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N inference results; 0 runs until Ctrl+C")
    parser.add_argument("--timeout-ms", type=int, default=15000, help="Fail if no inference result arrives within this interval")
    parser.add_argument("--summary", type=Path, help="Write final run statistics as JSON")
    parser.add_argument("--print-backend", action="store_true")
    if configure:
        configure(parser)
    args = parser.parse_args(argv)
    for name in ("model", "labels"):
        if not getattr(args, name).is_file():
            parser.error(f"{name} file does not exist: {getattr(args, name)}")
    for name in ("width", "height", "fps", "capture_buffers", "max_detections", "timeout_ms", "bitrate"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("NV12 width and height must be even")
    if args.capture_buffers > 128:
        parser.error("--capture-buffers must be at most 128")
    if args.frames < 0 or args.channel < 0:
        parser.error("--frames and --channel must be nonnegative")
    if not 0 <= args.score <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("--score and --nms-iou must be between 0 and 1")
    for base in (args.video_port_base, args.metadata_port_base):
        if not 1 <= base + args.channel <= 65535:
            parser.error("output port (base + channel) must be between 1 and 65535")
    if not args.host.strip():
        parser.error("--host must not be empty")
    return args


def make_graph(neat, args, *, include_frames=False):
    camera = neat.CameraInputOptions()
    if args.camera:
        camera.camera_name = args.camera
    camera.width, camera.height = args.width, args.height
    camera.framerate_num, camera.framerate_den = args.fps, 1
    camera.format = "NV12"
    camera.buffer_name = "camera0"
    camera.queue_depth = 2
    camera.allow_cpu_fallback = args.allow_cpu_fallback

    options = neat.ModelOptions()
    options.preprocess.kind = neat.InputKind.Image
    options.preprocess.enable = neat.AutoFlag.On
    options.preprocess.input_max_width = args.width
    options.preprocess.input_max_height = args.height
    options.preprocess.input_max_depth = 3
    options.preprocess.color_convert.input_format = neat.PreprocessColorFormat.NV12
    options.preprocess.color_convert.output_format = neat.PreprocessColorFormat.RGB
    options.preprocess.resize.enable = neat.AutoFlag.On
    options.preprocess.resize.width = 640
    options.preprocess.resize.height = 640
    options.preprocess.resize.mode = neat.ResizeMode.Letterbox
    options.preprocess.resize.pad_value = 114
    options.preprocess.preset = neat.NormalizePreset.COCO_YOLO
    options.advanced_execution.preprocess_target = "EV74"
    options.advanced_execution.postprocess_target = "EV74"
    options.decode_type = neat.BoxDecodeType.YoloV26
    options.score_threshold = args.score
    options.nms_iou_threshold = args.nms_iou
    options.top_k = args.max_detections
    model = neat.Model(str(args.model.resolve()), options)

    route = neat.ModelRouteOptions()
    route.upstream_name = camera.buffer_name
    route.buffer_name = camera.buffer_name
    route.name_suffix = "_camera0"

    source = neat.Graph("camera")
    source.add(neat.nodes.camera_input(camera, capture_buffer_count=args.capture_buffers))
    detector = neat.Graph("detector")
    detector.add(neat.nodes.input("inference"))
    detector.add(model.graph(route))
    detector.add(neat.nodes.output("detections", neat.OutputOptions.latest()))

    graph = neat.Graph("mipi_yolo26")
    branches = ["inference"] if args.no_stream else ["video", "inference"]
    if include_frames:
        branches.append("frame")
    split = neat.graphs.branch("camera", branches)
    graph.connect(source, split)
    if not args.no_stream:
        video_options = neat.VideoSenderOptions.h264_rtp_udp_from_raw(args.width, args.height, args.fps)
        video_options.host = args.host
        video_options.channel = args.channel
        video_options.video_port_base = args.video_port_base
        video_options.encoder.bitrate_kbps = args.bitrate
        video = neat.Graph("preview")
        video.add(neat.nodes.input("video"))
        video.add(neat.groups.video_sender(video_options))
        graph.connect(split, video)
    graph.connect(split, detector)
    if include_frames:
        frames = neat.Graph("frames")
        frames.add(neat.nodes.output("frame", neat.OutputOptions.latest()))
        graph.connect(split, frames)
    return graph, model


def detection_objects(neat, sample, args, labels):
    tensors = list(sample.tensors)
    if len(tensors) != 1:
        raise RuntimeError(f"expected one YOLO BBOX tensor, received {len(tensors)}")
    decoded = neat.detections.decode_bbox_tensor(tensors[0], args.width, args.height, strict=True)
    objects = []
    for box in decoded.boxes:
        if box.score < args.score or box.x2 <= box.x1 or box.y2 <= box.y1:
            continue
        if not 0 <= box.class_id < len(labels):
            raise RuntimeError(f"class {box.class_id} has no label")
        objects.append({"id": str(len(objects)), "label": labels[box.class_id],
                        "confidence": box.score,
                        "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1]})
    return objects


def detect(args):
    import pyneat as neat

    labels = [s.strip() for s in args.labels.read_text().splitlines() if s.strip()]
    if len(labels) != 80:
        raise ValueError(f"YOLO26 COCO model requires 80 labels, got {len(labels)}")
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True

    previous_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    run = None
    started = None
    summary = {"frames": 0, "detections": 0, "metadata_drops": 0, "error": None}
    classes = Counter()
    try:
        graph, model = make_graph(neat, args)
        sender = None
        if not args.no_stream:
            output = neat.MetadataSenderOptions()
            output.host, output.channel = args.host, args.channel
            output.metadata_port_base = args.metadata_port_base
            sender = neat.MetadataSender(output)
            print(f"Insight channel {args.channel}: video UDP {args.host}:{args.video_port_base + args.channel}, "
                  f"metadata UDP {args.host}:{args.metadata_port_base + args.channel}", flush=True)
        if args.print_backend:
            print(graph.describe_backend(False), flush=True)
        runtime = neat.RunOptions()
        runtime.preset = neat.RunPreset.Realtime
        runtime.queue_depth = 2
        runtime.overflow_policy = neat.OverflowPolicy.KeepLatest
        runtime.advanced.copy_input = False
        print(f"Starting {args.camera or 'default MIPI camera'} at {args.width}x{args.height}@{args.fps}; "
              f"model={args.model.name}; Ctrl+C stops", flush=True)
        run = graph.build(runtime)
        started = last_frame = last_report = time.monotonic()
        while not stopping and (not args.frames or summary["frames"] < args.frames):
            sample = run.pull("detections", min(1000, args.timeout_ms))
            now = time.monotonic()
            if sample is None:
                if now - last_frame >= args.timeout_ms / 1000:
                    raise RuntimeError(f"no detection output for {args.timeout_ms} ms: {run.last_error()}")
                continue
            last_frame = now
            objects = detection_objects(neat, sample, args, labels)
            summary["frames"] += 1
            summary["detections"] += len(objects)
            classes.update(obj["label"] for obj in objects)
            if sender is not None:
                # Both branches use the camera PTS; inference wall time cannot pair with video.
                if sample.pts_ns is None or sample.pts_ns < 0:
                    raise RuntimeError("detection output has no camera PTS for video pairing")
                sent = sender.send_metadata("object-detection", json.dumps({"objects": objects}),
                                            sample.pts_ns // 1_000_000, str(sample.frame_id))
                if not sent:
                    summary["metadata_drops"] += 1
            if summary["frames"] == 1 or now - last_report >= 1:
                names = ", ".join(f"{o['label']} {o['confidence']:.2f}" for o in objects) or "none"
                print(f"frame={summary['frames']} fps={summary['frames'] / max(now - started, .001):.1f} "
                      f"objects={len(objects)} [{names}]", flush=True)
                last_report = now
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        if run is not None:
            run.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        elapsed = time.monotonic() - started if started is not None else 0
        summary.update(elapsed_s=round(elapsed, 3),
                       fps=round(summary["frames"] / elapsed, 2) if elapsed else 0,
                       classes=dict(classes), stopped_by_signal=stopping)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY " + json.dumps(summary), flush=True)


def main(argv=None):
    args = arguments(argv)
    try:
        detect(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
