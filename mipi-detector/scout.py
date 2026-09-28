#!/usr/bin/env python3
"""SCOUT: one MIPI camera, YOLO tracking, Gemma verification and a live mission console."""

from collections import deque
from copy import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import gc
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
from scout_watch import WatchState, UsbWatch
from scout_config import load_config, configure_defaults
from scout_speech import Speech, client_id
from scout_lifecycle import RestartModels, shutdown_all

HERE = Path(__file__).resolve().parent


def configure(parser):
    parser.description = __doc__
    parser.add_argument('--config', type=Path, help='SCOUT YAML configuration; CLI options override it')
    parser.add_argument('--no-stt', action='store_true', help='Disable Whisper, including model loading')
    parser.add_argument('--no-tts', action='store_true', help='Disable spoken replies')
    parser.add_argument('--vlm-model', type=Path,
                        default=Path('/media/nvme/llima/models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy'))
    parser.add_argument('--vlm-timeout', type=float, default=15)
    parser.add_argument('--auto-checks', action=argparse.BooleanOptionalAction, default=True,
                        help='Automatically repeat subject checks (default: enabled)')
    parser.add_argument('--inspection-interval', type=float, default=3.0,
                        help='Minimum seconds between subject snapshot starts (default: 3)')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8022)
    parser.add_argument('--insight-vf-port', type=int, default=8081)
    parser.add_argument('--runtime-dir', type=Path, default=Path.home() / '.cache/neat-scout')
    parser.add_argument('--cert', type=Path)
    parser.add_argument('--key', type=Path)
    parser.add_argument('--mission', default='', help='Optional initial appearance description')
    parser.add_argument('--object-class', default='person')
    parser.add_argument('--watch-zone', action='store_true')
    parser.add_argument('--usb-camera', default=None, help='Optional USB /dev/v4l/by-id path, or auto')
    parser.add_argument('--usb-width', type=int, default=1280)
    parser.add_argument('--usb-height', type=int, default=720)
    parser.add_argument('--usb-fps', type=int, default=15, help='USB watch processing rate')


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
    return jpeg_image(crop)


def jpeg_image(crop):
    import cv2
    import numpy as np

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


class Preview:
    """Persistent encoder for the application lifetime."""

    def __init__(self, neat, args):
        self.neat = neat
        self.started_ns = time.monotonic_ns()
        self.runtime = None
        video = neat.VideoSenderOptions.h264_rtp_udp_from_raw(args.width, args.height, args.fps)
        video.host, video.channel = args.host, args.channel
        video.video_port_base = args.video_port_base
        video.encoder.bitrate_kbps = args.bitrate
        self.graph = neat.Graph('scout_preview')
        self.graph.connect(neat.nodes.input('video'), neat.groups.video_sender(video))
        self.options = neat.RunOptions()
        self.options.preset = neat.RunPreset.Realtime
        self.options.queue_depth = 2
        self.options.overflow_policy = neat.OverflowPolicy.KeepLatest
        self.options.advanced.copy_input = False
        self.options.startup_preflight = False

    def send(self, frame, pts_ns):
        tensor = frame.tensor if frame.tensor is not None else frame.tensors[0]
        # A separate header shares the pixels without changing capture timestamps
        # used to match detections and evidence in the camera run.
        sample = self.neat.make_tensor_sample('video', tensor)
        sample.caps_string = frame.caps_string
        sample.pts_ns, sample.frame_id = pts_ns, frame.frame_id
        sample.duration_ns = frame.duration_ns
        if self.runtime is None:
            self.runtime = self.graph.build([sample], self.options)
        else:
            self.runtime.push('video', [sample])

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None


class FrameCache:
    """Drain raw output independently; retain at most eight of the 32 camera buffers."""

    def __init__(self, runtime, preview=None):
        self.runtime = runtime
        self.preview = preview
        self.pts_offset_ns = None
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
                    if self.preview:
                        if sample.pts_ns is None or sample.pts_ns < 0:
                            raise RuntimeError('Missing camera timestamp for preview')
                        if self.pts_offset_ns is None:
                            # Camera PTS resets on restart; preserve one monotonic
                            # video timeline, with the same offset for metadata.
                            self.pts_offset_ns = (time.monotonic_ns() -
                                self.preview.started_ns - sample.pts_ns)
                        self.preview.send(sample, sample.pts_ns + self.pts_offset_ns)
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
    def __init__(self, args, state, vlm, speech=None):
        self.args, self.state, self.vlm = args, state, vlm
        self.speech = speech
        self.owners = {'mipi': None, 'usb': None}
        self.retry_vlm = threading.Event()
        self.closing = False
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
                          '/voice.js': (HERE / 'scout-web/voice.js', 'text/javascript'),
                          '/audio-recorder.js': (HERE / 'scout-web/audio-recorder.js', 'text/javascript'),
                          '/webrtc.js': (detector.ROOT / 'web/webrtc.js', 'text/javascript')}
                if path in assets:
                    filename, mime = assets[path]
                    self.send(200, filename.read_bytes(), mime)
                elif path == '/api/state':
                    snapshot = console.state.snapshot()
                    snapshot['speech'] = console.speech.snapshot()
                    self.send(200, snapshot)
                elif path == '/api/speech/state':
                    owner = parse_qs(urlsplit(self.path).query).get('client_id', [''])[0]
                    self.send(200, console.speech.snapshot(owner))
                elif path.startswith('/api/speech/audio/'):
                    parts = path.split('/')
                    payload = console.speech.audio(parts[4], parts[5]) if len(parts) == 6 else None
                    self.send(200, payload, 'audio/wav') if payload else self.send(404, {'error': 'Audio expired'})
                elif path == '/api/config':
                    self.send(200, dict(channel=console.args.channel, width=console.args.width,
                        height=console.args.height, requested_fps=console.args.fps,
                        classes=console.state.labels, model='Gemma 4 E4B', source='MIPI 01',
                        speech=console.speech.public_config(),
                        usb=dict(channel=console.args.channel+1, width=console.args.usb_width,
                                 height=console.args.usb_height, source='USB 02') if console.args.usb_camera else None))
                elif path.startswith('/api/evidence/') and path.endswith('.jpg'):
                    key = path.rsplit('/', 1)[-1][:-4]
                    with console.state.lock:
                        payload = console.state.evidence.get(key)
                    self.send(200, payload, 'image/jpeg') if payload else self.send(404, {'error': 'Evidence expired'})
                else:
                    self.send(404, {'error': 'Not found'})

            def do_POST(self):
                try:
                    if console.closing:
                        self.send(503, {'error': 'All models are stopping. Reconnect after restart.'})
                        return
                    origin = self.headers.get('Origin')
                    if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                        self.send(403, {'error': 'Use the mission console on this server.'})
                        return
                    size = int(self.headers.get('Content-Length', '0'))
                    path = urlsplit(self.path).path
                    if path.startswith('/api/speech/recordings/'):
                        if not 0 < size <= console.speech.settings['stt']['upload_max_bytes']:
                            raise ValueError('Invalid recording size')
                        if self.headers.get_content_type() != 'audio/wav':
                            raise ValueError('Expected audio/wav')
                        self.connection.settimeout(15)
                        console.speech.upload(path.rsplit('/', 1)[-1], self.rfile.read(size))
                        self.send(202, {'status': 'queued'})
                        return
                    if not 0 < size <= 262144:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict):
                        raise ValueError('Expected a JSON object')
                    path = urlsplit(self.path).path
                    if path == '/offer':
                        channel = parse_qs(urlsplit(self.path).query).get('channel', [''])[0]
                        channels = [str(console.args.channel)]
                        if console.args.usb_camera:
                            channels.append(str(console.args.channel+1))
                        if channel not in channels:
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
                        if set(body) - {'query', 'object_class', 'zone', 'client_id'}:
                            raise ValueError('Unknown mission fields')
                        owner = client_id(body['client_id']) if 'client_id' in body else None
                        with console.state.lock:
                            console.state.start(body.get('query'), body.get('object_class'), body.get('zone', False))
                            console.speech.cancel(camera='mipi')
                            console.owners['mipi'] = owner
                            if not console.vlm.active or console.vlm.active.get('camera_id') != 'usb':
                                console.vlm.cancel()
                        self.send(200, console.state.snapshot())
                    elif path == '/api/reset':
                        with console.state.lock:
                            console.state.reset()
                            console.speech.cancel(camera='mipi')
                            console.owners['mipi'] = None
                            if not console.vlm.active or console.vlm.active.get('camera_id') != 'usb':
                                console.vlm.cancel()
                        self.send(200, console.state.snapshot())
                    elif path == '/api/watch':
                        if set(body) - {'enabled', 'object_class', 'zone'}:
                            raise ValueError('Unknown watch fields')
                        with console.state.lock:
                            watch = console.state.watch
                            if watch is None:
                                raise ValueError('USB watch is not configured')
                            watch.configure(body.get('enabled', watch.enabled),
                                            body.get('object_class', watch.object_class),
                                            body.get('zone', watch.zone))
                            console.speech.cancel(camera='usb')
                            console.owners['usb'] = None
                            console.state.event('mission', 'Watch area updated',
                                ('Watching ' + ('people and payload props' if watch.object_class == 'any' else watch.object_class))
                                if watch.enabled else 'Area monitoring stopped.',
                                camera_id='usb', camera_label='USB 02')
                            if console.vlm.active and console.vlm.active.get('camera_id') == 'usb':
                                console.vlm.cancel()
                        self.send(200, console.state.snapshot())
                    elif path == '/api/watch/inspect':
                        owner = client_id(body['client_id']) if 'client_id' in body else None
                        with console.state.lock:
                            watch = console.state.watch
                            if watch is None or watch.snapshot()['status'] != 'live':
                                raise ValueError('Wait for a live USB view')
                            if console.state.vlm_status != 'ready' or console.state.busy:
                                raise ValueError('Wait for Gemma to be ready')
                            watch.inspection_requested = True
                            console.owners['usb'] = owner
                        self.send(202, console.state.snapshot())
                    elif path == '/api/speech/begin':
                        with console.state.lock:
                            if body.get('object_class') not in console.state.labels or type(body.get('zone')) is not bool:
                                raise ValueError('Choose a supported object class and zone setting')
                            token = console.speech.begin(body.get('client_id'), console.state.generation + 1,
                                                         body['object_class'], body['zone'])
                            console.state.generation += 1
                            console.state.focus = None
                            console.state.checked_at.clear()
                            console.state.track_verdicts.clear()
                            console.state.inspection_requested = False
                            console.owners['mipi'] = None
                            if not console.vlm.active or console.vlm.active.get('camera_id') != 'usb':
                                console.vlm.cancel()
                        self.send(201, {'id': token, 'mission_id': console.state.generation})
                    elif path == '/api/speech/cancel':
                        owner = client_id(body.get('client_id'))
                        with console.state.lock:
                            console.speech.cancel(owner=owner)
                            for camera, current in console.owners.items():
                                if current == owner:
                                    console.owners[camera] = None
                        self.send(200, {'status': 'cancelled'})
                    elif path == '/api/vlm/retry':
                        console.retry_vlm.set()
                        self.send(202, {'status': 'All models will stop and restart together'})
                    else:
                        self.send(404, {'error': 'Not found'})
                except (ValueError, TypeError, TimeoutError) as exc:
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

    def request_stop(self):
        self.closing = True

    def close(self):
        self.request_stop()
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
    state = MissionState(labels, auto_checks=args.auto_checks, inspection_interval=args.inspection_interval)
    if args.usb_camera:
        state.watch = WatchState(labels)
    vlm = VLMWorker(args.vlm_model, args.vlm_timeout)
    speech = Speech(args.speech_config)
    console = Console(args, state, vlm, speech)
    stopping = threading.Event()
    handlers = {s: signal.signal(s, lambda *_: stopping.set())
                for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    runtime = cache = graph = model = preview = usb = None
    capture_args = copy(args)
    capture_args.no_stream = True
    args.runtime_dir.mkdir(parents=True, exist_ok=True)

    try:
        console.start()
        print(f'SCOUT console: https://{socket.gethostname()}:{args.port}', flush=True)
        # Initialize the complete resident group before starting the camera runs.
        # No member is unloaded or restarted independently during the session.
        validate_model(args.vlm_model)
        vlm.start()
        while not stopping.is_set() and state.vlm_status == 'loading':
            for message in vlm.poll():
                if message['kind'] == 'ready':
                    state.vlm_status = 'ready'
                    print('Gemma ready: ' + message['model'], flush=True)
                elif message['kind'] == 'fatal':
                    raise RuntimeError(message['error'])
            stopping.wait(.1)
        if not stopping.is_set():
            speech.start_models()
            while not stopping.is_set() and not speech.poll_models():
                for message in vlm.poll():
                    if message['kind'] == 'fatal':
                        raise RuntimeError(message['error'])
                stopping.wait(.1)
        if stopping.is_set():
            return
        if not args.no_stream:
            preview = Preview(neat, args)
        graph, model = detector.make_graph(neat, capture_args, include_frames=True)
        options = neat.RunOptions()
        options.preset = neat.RunPreset.Realtime
        options.queue_depth = 2
        options.overflow_policy = neat.OverflowPolicy.KeepLatest
        options.output_memory = neat.OutputMemory.ZeroCopy
        options.advanced.copy_input = False
        if args.print_backend:
            print(graph.describe_backend(False), flush=True)
        runtime = graph.build(options)
        cache = FrameCache(runtime, preview)
        if args.usb_camera:
            usb = UsbWatch(neat, args, state, Preview, jpeg_image)
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
                raise RestartModels('User requested a complete model-group restart')
            speech.poll_models()
            if cache.error:
                raise RuntimeError('Camera frame output failed: ' + cache.error)
            if usb and (usb.error or not usb.thread.is_alive()):
                raise RuntimeError('USB model worker stopped: ' + (usb.error or 'unexpected exit'))
            for message in vlm.poll():
                if message['kind'] == 'ready':
                    state.vlm_status = 'ready'
                    print('Gemma ready: ' + message['model'], flush=True)
                elif message['kind'] == 'fatal':
                    raise RuntimeError(message['error'])
                elif message['kind'] == 'result' and vlm.active:
                    job = vlm.active
                    vlm.active = None
                    with state.lock:
                        before = state.event_counter
                        camera = job.get('camera_id', 'mipi')
                        if camera == 'usb':
                            usb.apply_result(job, message)
                        else:
                            state.apply_snapshot_result(job, message, time.monotonic())
                        if (state.event_counter != before and not message.get('error')
                                and state.events[0].get('announce', True)):
                            speech.speak(console.owners[camera], state.events[0], job['generation'])
                    print('VERIFICATION ' + json.dumps({k: v for k, v in message.items()}), flush=True)
            completed = speech.poll()
            if completed and completed['kind'] == 'stt' and not completed.get('cancelled') and not completed.get('error'):
                with state.lock, speech.lock:
                    if not completed.get('cancelled') and completed['generation'] == state.generation:
                        try:
                            state.start(completed.get('transcript'), completed['object_class'], completed['zone'])
                            completed['mission_id'] = state.generation
                            console.owners['mipi'] = completed['client_id']
                        except (ValueError, TypeError) as exc:
                            completed['error'] = str(exc)
                            speech.phase = 'error'
                    else:
                        speech.cancel()
            if speech.phase == 'queued':
                speech.start_pending()
            if usb and not state.busy and state.vlm_status == 'ready':
                job = usb.inspection_job()
                if job:
                    state.busy = True
                    state.vlm_requests += 1
                    vlm.submit(job)
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
                            focus=dict(state.focus) if state.focus else None,
                            zone=dict(enabled=state.zone_enabled, inside=state.zone_inside, bbox=state.zone))
            if sender and cache.pts_offset_ns is not None:
                if sample.pts_ns is None or sample.pts_ns < 0:
                    raise RuntimeError('Missing camera timestamp')
                sender.send_metadata('object-detection', json.dumps(meta),
                                     (sample.pts_ns + cache.pts_offset_ns) // 1_000_000,
                                     str(sample.frame_id))
            with state.lock:
                voice_mission_pending = (speech.job and speech.job['kind'] == 'stt' and speech.busy)
                candidate = None if voice_mission_pending else state.choose_candidate(now)
                frame = cache.find(sample.pts_ns) if candidate else None
                if candidate and frame is not None:
                    jpeg = evidence_image(frame, candidate.bbox, args.width, args.height)
                    job = dict(generation=state.generation, query=state.query, object_class=state.object_class,
                               track_id=candidate.id, bbox=list(candidate.bbox), pts_ms=sample.pts_ns // 1_000_000,
                               captured_at=now, captured_time=time.time(), jpeg=jpeg)
                    state.begin_inspection(candidate.id, now)
            # Never retain a camera sample while waiting for Gemma.
            sample = frame = None
            if job is not None:
                # Evidence is a copied JPEG. Capture, YOLO and encoders stay live.
                vlm.submit(job)
            if now - last_report >= 2:
                print(f'frames={state.frames} fps={state.fps:.1f} tracks={len(tracks)} '
                      f'mission={state.phase} vlm={state.vlm_status} busy={state.busy}', flush=True)
                last_report = now
    except RestartModels:
        state.camera_status = 'restarting'
        raise
    except Exception as exc:
        state.camera_status, state.camera_error = 'error', str(exc)
        raise
    finally:
        print('SCOUT: stopping the complete model group', flush=True)
        # Signal all owners first. Even a failed close must not skip another model.
        resources = [('console', console.request_stop, console.close),
                     ('Gemma', vlm.request_stop, vlm.close),
                     ('speech', speech.request_stop, speech.close)]
        if usb is not None:
            resources.append(('USB', usb.request_stop, usb.close))
        if cache is not None:
            resources.append(('MIPI frames', cache.stopping.set, cache.close))
        if runtime is not None:
            resources.append(('MIPI camera/YOLO', None, runtime.close))
        if preview is not None:
            resources.append(('MIPI preview', None, preview.close))
        cleanup_errors = shutdown_all(resources)
        # Release graph/model/sample references and their device allocations too.
        resources.clear()
        sample = frame = sender = runtime = cache = graph = model = preview = usb = None
        gc.collect()
        for error in cleanup_errors:
            print('SCOUT cleanup error: ' + error, file=sys.stderr, flush=True)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if state.camera_status not in ('error', 'restarting'):
            state.camera_status = 'stopped'
        state.busy = False
        state.vlm_status = 'stopped'
        summary = state.snapshot()
        summary['speech'] = speech.snapshot()
        summary['cleanup_errors'] = cleanup_errors
        (args.runtime_dir / 'last-run.json').write_text(json.dumps(summary, indent=2) + '\n')
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + '\n')
        print('SCOUT SUMMARY ' + json.dumps(summary), flush=True)
        if cleanup_errors:
            raise RuntimeError('Model-group cleanup failed; automatic restart cancelled: ' + '; '.join(cleanup_errors))


def main(argv=None):
    defaults, speech_config = load_config(argv)
    def configured(parser):
        configure(parser)
        configure_defaults(parser, defaults)
    args = detector.arguments(argv, configure=configured)
    args.speech_config = speech_config
    if not 1 <= args.port <= 65535 or not 1 <= args.insight_vf_port <= 65535:
        raise ValueError('Ports must be between 1 and 65535')
    if not 1 <= args.vlm_timeout <= 60:
        raise ValueError('Use a VLM timeout of 1–60 seconds')
    if not 1 <= args.inspection_interval <= 300:
        raise ValueError('Use an inspection interval of 1–300 seconds')
    if args.usb_camera:
        if args.usb_width < 160 or args.usb_height < 120 or args.usb_width % 2 or args.usb_height % 2:
            raise ValueError('USB dimensions must be even and at least 160×120')
        if not 1 <= args.usb_fps <= 30:
            raise ValueError('USB FPS must be 1–30')
        if any(base + args.channel + 1 > 65535 for base in (args.video_port_base, args.metadata_port_base)):
            raise ValueError('USB channel port is out of range')
    if bool(args.cert) != bool(args.key):
        raise ValueError('Supply both --cert and --key')
    if not (args.vlm_model / 'devkit/vlm_config.json').is_file():
        raise ValueError(f'Missing VLM model: {args.vlm_model}')
    try:
        run(args)
    except RestartModels:
        # Native device descriptors may survive execv. Fully exit this process;
        # scout.sh waits for it before starting a fresh model group/Python PID.
        print('SCOUT: all models closed; exiting for a complete restart', flush=True)
        return 75
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'SCOUT error: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
