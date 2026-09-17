#!/usr/bin/env python3
"""SCOUT: one MIPI camera, YOLO tracking, Gemma verification and a live mission console."""

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import main as detector
from scout_state import MissionState
from scout_vlm import VLMWorker, validate_model

HERE = Path(__file__).resolve().parent


def configure(parser):
    parser.description = __doc__
    parser.add_argument('--vlm-model', type=Path,
                        default=Path('/workspace/llima/models/gemma-4-E4B-it-GPTQ-a16w4'))
    parser.add_argument('--vlm-timeout', type=float, default=15)
    parser.add_argument('--auto-checks', action='store_true',
                        help='Repeat snapshot checks, with at least 15 seconds of live capture between checks')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8022)
    parser.add_argument('--insight-vf-port', type=int, default=8081)
    parser.add_argument('--runtime-dir', type=Path, default=Path.home() / '.cache/neat-scout')
    parser.add_argument('--cert', type=Path)
    parser.add_argument('--key', type=Path)
    parser.add_argument('--mission', default='', help='Optional initial appearance description')
    parser.add_argument('--object-class', default='person')
    parser.add_argument('--watch-zone', action='store_true')


def evidence_image(frame, bbox, width, height):
    """Copy only a requested observation, then make the exact image Gemma will see."""
    import cv2
    import numpy as np

    tensor = frame.tensor if frame.tensor is not None else frame.tensors[0]
    if not tensor.is_nv12() or tensor.width() != width or tensor.height() != height:
        raise RuntimeError('Expected a matching NV12 camera frame for verification')
    payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    if payload.size != expected:
        raise RuntimeError(f'Unexpected NV12 layout: {payload.size} bytes, expected {expected}')
    bgr = cv2.cvtColor(payload.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    x, y, w, h = bbox
    x1, y1 = max(0, int(x - w * .08)), max(0, int(y - h * .08))
    x2, y2 = min(width, int(x + w * 1.08)), min(height, int(y + h * 1.08))
    crop = bgr[y1:y2, x1:x2]
    if not crop.size:
        raise RuntimeError('Empty candidate crop')
    scale = min(480 / crop.shape[1], 480 / crop.shape[0])
    resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                               max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    square = np.full((480, 480, 3), 32, dtype=np.uint8)
    oy, ox = (480 - resized.shape[0]) // 2, (480 - resized.shape[1]) // 2
    square[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    ok, jpeg = cv2.imencode('.jpg', square, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError('Could not encode verification evidence')
    return jpeg.tobytes()


class FrameCache:
    """Drain raw output independently; retain at most eight of the 32 camera buffers."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.frames = deque(maxlen=8)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stopping.is_set():
                sample = self.runtime.pull('frame', 200)
                if sample is not None:
                    with self.lock:
                        self.frames.append(sample)
        except Exception as exc:
            self.error = str(exc)

    def find(self, pts_ns):
        with self.lock:
            return next((s for s in reversed(self.frames) if s.pts_ns == pts_ns), None)

    def close(self):
        self.stopping.set()
        self.thread.join(2)
        with self.lock:
            self.frames.clear()


class Console:
    def __init__(self, args, state, vlm):
        self.args, self.state, self.vlm = args, state, vlm
        self.retry_vlm = threading.Event()
        self.server = None
        self.thread = None

    def start(self):
        console = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, status, payload, content_type='application/json'):
                if isinstance(payload, (dict, list)):
                    payload = json.dumps(payload).encode()
                elif isinstance(payload, str):
                    payload = payload.encode()
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlsplit(self.path).path
                assets = {'/': (HERE / 'scout-web/index.html', 'text/html; charset=utf-8'),
                          '/scout.css': (HERE / 'scout-web/scout.css', 'text/css'),
                          '/scout.js': (HERE / 'scout-web/scout.js', 'text/javascript'),
                          '/webrtc.js': (detector.ROOT / 'web/webrtc.js', 'text/javascript')}
                if path in assets:
                    filename, mime = assets[path]
                    self.send(200, filename.read_bytes(), mime)
                elif path == '/api/state':
                    self.send(200, console.state.snapshot())
                elif path == '/api/config':
                    self.send(200, dict(channel=console.args.channel, width=console.args.width,
                        height=console.args.height, requested_fps=console.args.fps,
                        classes=console.state.labels, model='Gemma 4 E4B', source='MIPI 01'))
                elif path.startswith('/api/evidence/') and path.endswith('.jpg'):
                    key = path.rsplit('/', 1)[-1][:-4]
                    with console.state.lock:
                        payload = console.state.evidence.get(key)
                    self.send(200, payload, 'image/jpeg') if payload else self.send(404, {'error': 'Evidence expired'})
                else:
                    self.send(404, {'error': 'Not found'})

            def do_POST(self):
                try:
                    origin = self.headers.get('Origin')
                    if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                        self.send(403, {'error': 'Use the mission console on this server.'})
                        return
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 262144:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict):
                        raise ValueError('Expected a JSON object')
                    path = urlsplit(self.path).path
                    if path == '/offer':
                        channel = parse_qs(urlsplit(self.path).query).get('channel', [''])[0]
                        if channel != str(console.args.channel):
                            raise ValueError('Unknown camera channel')
                        url = f'https://{console.args.host}:{console.args.insight_vf_port}/offer?channel={channel}'
                        req = Request(url, data=json.dumps(body).encode(),
                                      headers={'Content-Type': 'application/json'})
                        try:
                            with urlopen(req, context=ssl._create_unverified_context(), timeout=12) as response:
                                self.send(response.status, response.read())
                        except HTTPError as exc:
                            self.send(exc.code, exc.read())
                        except (URLError, TimeoutError) as exc:
                            self.send(502, {'error': f'Insight connection failed: {exc}'})
                    elif path == '/api/mission':
                        if set(body) - {'query', 'object_class', 'zone'}:
                            raise ValueError('Unknown mission fields')
                        with console.state.lock:
                            console.state.start(body.get('query'), body.get('object_class'), body.get('zone', False))
                            console.vlm.cancel()
                        self.send(200, console.state.snapshot())
                    elif path == '/api/reset':
                        with console.state.lock:
                            console.state.reset()
                            console.vlm.cancel()
                        self.send(200, console.state.snapshot())
                    elif path == '/api/vlm/retry':
                        console.retry_vlm.set()
                        self.send(202, {'status': 'Retry requested'})
                    else:
                        self.send(404, {'error': 'Not found'})
                except (ValueError, TypeError) as exc:
                    self.send(400, {'error': str(exc)})

        cert, key = self.args.cert, self.args.key
        if not cert:
            cert, key = self.args.runtime_dir / 'cert.pem', self.args.runtime_dir / 'key.pem'
            if not cert.exists() or not key.exists():
                subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '365',
                    '-subj', '/CN=modalix-scout', '-addext', 'subjectAltName=DNS:modalix,DNS:localhost'],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                key.chmod(0o600)
        self.server = ThreadingHTTPServer((self.args.bind, self.args.port), Handler)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(cert, key)
        self.server.socket = tls.wrap_socket(self.server.socket, server_side=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(2)


def run(args):
    import cv2
    import numpy  # Load image-copy dependencies before starting live capture.
    import pyneat as neat

    cv2.setNumThreads(1)
    labels = [s.strip() for s in args.labels.read_text().splitlines() if s.strip()]
    if len(labels) != 80:
        raise ValueError('The supplied YOLO26 model needs 80 COCO labels')
    state = MissionState(labels, auto_checks=args.auto_checks)
    vlm = VLMWorker(args.vlm_model, args.vlm_timeout)
    console = Console(args, state, vlm)
    stopping = threading.Event()
    handlers = {s: signal.signal(s, lambda *_: stopping.set())
                for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    runtime = cache = None
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        console.start()
        print(f'SCOUT console: https://{socket.gethostname()}:{args.port}', flush=True)
        # Load Gemma before starting capture; loading a large model while video
        # is live can interrupt the camera/MLA pipeline on this runtime.
        try:
            validate_model(args.vlm_model)
            vlm.start()
            deadline = time.monotonic() + 180
            while not stopping.is_set() and state.vlm_status == 'loading':
                for message in vlm.poll():
                    if message['kind'] == 'ready':
                        state.vlm_status = 'ready'
                        print('Gemma ready: ' + message['model'], flush=True)
                    elif message['kind'] == 'fatal':
                        state.vlm_status, state.vlm_error = 'error', message['error']
                        vlm.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError('Gemma model loading exceeded 180 seconds')
                stopping.wait(.1)
        except Exception as exc:
            state.vlm_status, state.vlm_error = 'error', str(exc)
            vlm.close()
        if stopping.is_set():
            return
        graph, model = detector.make_graph(neat, args, include_frames=True)
        options = neat.RunOptions()
        options.preset = neat.RunPreset.Realtime
        options.queue_depth = 2
        options.overflow_policy = neat.OverflowPolicy.KeepLatest
        options.output_memory = neat.OutputMemory.ZeroCopy
        options.advanced.copy_input = False
        if args.print_backend:
            print(graph.describe_backend(False), flush=True)
        runtime = graph.build(options)
        cache = FrameCache(runtime)
        meta_options = neat.MetadataSenderOptions()
        meta_options.host, meta_options.channel = args.host, args.channel
        meta_options.metadata_port_base = args.metadata_port_base
        sender = neat.MetadataSender(meta_options) if not args.no_stream else None
        if args.mission:
            state.start(args.mission, args.object_class, args.watch_zone)
        last_frame = last_report = time.monotonic()
        frame_times = deque(maxlen=100)
        while not stopping.is_set() and (not args.frames or state.frames < args.frames):
            if console.retry_vlm.is_set():
                console.retry_vlm.clear()
                state.pause_capture()
                if cache:
                    cache.close()
                if runtime:
                    runtime.close()
                cache = runtime = None
                vlm.close()
                with state.lock:
                    state.busy = False
                    state.candidate = None
                    state.vlm_error = None
                    state.vlm_status = 'loading'
                try:
                    validate_model(args.vlm_model)
                    vlm.start()
                except Exception as exc:
                    state.vlm_status, state.vlm_error = 'error', str(exc)
            for message in vlm.poll():
                if message['kind'] == 'ready':
                    state.vlm_status = 'ready'
                    print('Gemma ready: ' + message['model'], flush=True)
                elif message['kind'] == 'fatal':
                    with state.lock:
                        state.vlm_status, state.vlm_error = 'error', message['error']
                        state.busy = False
                        state.candidate = None
                        if state.query:
                            state.phase = 'searching'
                    print('VLM error: ' + message['error'], flush=True)
                    vlm.close()
                elif message['kind'] == 'result' and vlm.active:
                    job = vlm.active
                    vlm.active = None
                    state.apply_snapshot_result(job, message, time.monotonic())
                    print('VERIFICATION ' + json.dumps({k: v for k, v in message.items()}), flush=True)
            if runtime is None:
                if state.busy or state.vlm_status == 'loading':
                    stopping.wait(.05)
                    continue
                state.camera_status = 'resuming'
                runtime = graph.build(options)
                cache = FrameCache(runtime)
                frame_times.clear()
                last_frame = time.monotonic()
                continue
            sample = runtime.pull('detections', 250)
            now = time.monotonic()
            if sample is None:
                if now - last_frame > args.timeout_ms / 1000:
                    raise RuntimeError(f'No camera observations: {runtime.last_error()}')
                continue
            last_frame = now
            if cache.error:
                raise RuntimeError('Camera frame output failed: ' + cache.error)
            objects = detector.detection_objects(neat, sample, args, labels)
            tracks = state.observe(objects, now, args.width, args.height)
            # Use capture timestamps so draining a short startup queue does not
            # briefly report more FPS than the sensor actually delivered.
            captured = sample.pts_ns / 1_000_000_000 if sample.pts_ns is not None else now
            frame_times.append(captured)
            if len(frame_times) > 1:
                state.fps = (len(frame_times) - 1) / max(.001, captured - frame_times[0])
            job = None
            with state.lock:
                meta = dict(objects=tracks, mission_id=state.generation, phase=state.phase,
                            candidate=state.candidate,
                            zone=dict(enabled=state.zone_enabled, inside=state.zone_inside, bbox=state.zone))
            if sender:
                if sample.pts_ns is None or sample.pts_ns < 0:
                    raise RuntimeError('Missing camera timestamp')
                sender.send_metadata('object-detection', json.dumps(meta),
                                     sample.pts_ns // 1_000_000, str(sample.frame_id))
            with state.lock:
                candidate = state.choose_candidate(now)
                frame = cache.find(sample.pts_ns) if candidate else None
                if candidate and frame is not None:
                    jpeg = evidence_image(frame, candidate.bbox, args.width, args.height)
                    job = dict(generation=state.generation, query=state.query, object_class=state.object_class,
                               track_id=candidate.id, bbox=list(candidate.bbox), pts_ms=sample.pts_ns // 1_000_000,
                               captured_at=now, captured_time=time.time(), jpeg=jpeg)
                    state.candidate = candidate.id
                    state.busy = True
                    state.inspection_requested = False
                    state.vlm_requests += 1
                    state.phase = 'verifying'
            # Never retain a camera sample while waiting for Gemma.
            sample = frame = None
            if job is not None:
                # This board's LLiMa inference overflows the active MIPI receiver.
                # Stop capture AND drain YOLO before handing the MLA to Gemma.
                # A result describes only the saved snapshot, not a future track.
                state.pause_capture()
                cache.close()
                runtime.close()
                cache = runtime = None
                vlm.submit(job)
            if now - last_report >= 2:
                print(f'frames={state.frames} fps={state.fps:.1f} tracks={len(tracks)} '
                      f'mission={state.phase} vlm={state.vlm_status} busy={state.busy}', flush=True)
                last_report = now
    except Exception as exc:
        state.camera_status, state.camera_error = 'error', str(exc)
        raise
    finally:
        vlm.close()
        if cache is not None:
            cache.close()
        if runtime is not None:
            runtime.close()
        console.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if state.camera_status != 'error':
            state.camera_status = 'stopped'
        summary = state.snapshot()
        (args.runtime_dir / 'last-run.json').write_text(json.dumps(summary, indent=2) + '\n')
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + '\n')
        print('SCOUT SUMMARY ' + json.dumps(summary), flush=True)


def main(argv=None):
    args = detector.arguments(argv, configure=configure)
    if not 1 <= args.port <= 65535 or not 1 <= args.insight_vf_port <= 65535:
        raise ValueError('Ports must be between 1 and 65535')
    if not 1 <= args.vlm_timeout <= 60:
        raise ValueError('Use a VLM timeout of 1–60 seconds')
    if bool(args.cert) != bool(args.key):
        raise ValueError('Supply both --cert and --key')
    if not (args.vlm_model / 'devkit/vlm_config.json').is_file():
        raise ValueError(f'Missing VLM model: {args.vlm_model}')
    run(args)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'SCOUT error: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
