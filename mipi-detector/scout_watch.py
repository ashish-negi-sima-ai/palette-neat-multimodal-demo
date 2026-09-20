"""USB landing/payload observations, independent of MIPI subject identities."""

from collections import deque
from copy import copy
import json
import math
from pathlib import Path
import threading
import time

import main as detector
from scout_state import Tracker

PAYLOAD_CLASSES = frozenset({'person', 'backpack', 'handbag', 'suitcase', 'bottle',
                            'cup', 'bowl', 'laptop', 'cell phone', 'book', 'chair'})


class WatchState:
    """All callers hold the parent mission lock; unknown never means clear."""

    def __init__(self, labels):
        self.labels = labels
        self.enabled = True
        self.object_class = 'any'
        self.zone = [.15, .15, .7, .7]
        self.generation = 0
        self.status, self.error = 'starting', None
        self.occupied = None
        self.pending = None
        self.pending_since = 0
        self.tracker = Tracker(prefix='U')
        self.objects = []
        self.frames, self.fps, self.last_frame_at = 0, 0., 0.
        self.inspection_requested = False
        self.checking = False
        self.last_verdict = None

    def configure(self, enabled, object_class, zone):
        if not isinstance(enabled, bool):
            raise ValueError('Watch enabled must be a boolean.')
        if object_class != 'any' and object_class not in self.labels:
            raise ValueError('Choose a supported watch class.')
        if (not isinstance(zone, list) or len(zone) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in zone)):
            raise ValueError('Watch area must contain four finite numbers.')
        x, y, w, h = zone
        if x < 0 or y < 0 or w < .05 or h < .05 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError('Draw an area inside the image, at least 5% wide and tall.')
        self.enabled, self.object_class, self.zone = enabled, object_class, list(zone)
        self.generation += 1
        self.occupied = self.pending = None
        self.inspection_requested = False
        self.last_verdict = None

    def unavailable(self, status, error=None):
        self.status, self.error = status, error
        self.occupied = self.pending = None
        self.objects = []
        self.tracker.tracks.clear()

    def observe(self, objects, now, width, height):
        tracks = self.tracker.update(objects, now)
        zx, zy, zw, zh = self.zone
        zone = [zx * width, zy * height, zw * width, zh * height]
        self.objects = []
        for track in tracks:
            x, y, w, h = track.bbox
            intersection = (max(0, min(x+w, zone[0]+zone[2])-max(x, zone[0])) *
                            max(0, min(y+h, zone[1]+zone[3])-max(y, zone[1])))
            watched = track.label in PAYLOAD_CLASSES if self.object_class == 'any' else track.label == self.object_class
            inside = (self.enabled and watched
                      and intersection / max(1, w*h) >= .2)
            self.objects.append(dict(id=track.id, label=track.label, bbox=track.bbox,
                                     confidence=track.confidence, inside=inside))
        self.frames += 1
        self.status, self.error, self.last_frame_at = 'live', None, now
        if not self.enabled:
            self.occupied = self.pending = None
            return None
        occupied = any(o['inside'] for o in self.objects)
        if occupied != self.pending:
            self.pending, self.pending_since = occupied, now
        settle = .5 if occupied else 1.0
        if now - self.pending_since >= settle and occupied != self.occupied:
            self.occupied = occupied
            return dict(occupied=occupied, labels=sorted({o['label'] for o in self.objects if o['inside']}))
        return None

    def snapshot(self):
        age = time.monotonic() - self.last_frame_at if self.last_frame_at else None
        live = self.status == 'live' and age is not None and age < 2
        return dict(enabled=self.enabled, object_class=self.object_class, zone=list(self.zone),
                    generation=self.generation, occupied=self.occupied if live else None,
                    status=self.status if self.status != 'live' or live else 'stale', error=self.error,
                    frames=self.frames, fps=round(self.fps, 1),
                    observation_age_s=round(age, 2) if age is not None else None,
                    objects=list(self.objects) if live else [], checking=self.checking,
                    inspection_requested=self.inspection_requested, last_verdict=self.last_verdict)


def resolve_camera(device):
    if device != 'auto':
        path = Path(device)
        if not path.exists():
            raise RuntimeError(f'USB camera not found: {device}')
        return str(path.resolve())
    candidates = sorted(Path('/dev/v4l/by-id').glob('*-video-index0'))
    if len(candidates) != 1:
        raise RuntimeError(f'Found {len(candidates)} USB camera identities; specify --usb-camera /dev/v4l/by-id/…-video-index0')
    return str(candidates[0].resolve())


class UsbWatch:
    def __init__(self, neat, args, state, preview_factory, jpeg_image):
        self.neat, self.state, self.jpeg_image = neat, state, jpeg_image
        self.args = copy(args)
        self.args.width, self.args.height = args.usb_width, args.usb_height
        self.args.fps, self.args.channel = args.usb_fps, args.channel + 1
        self.args.bitrate = min(args.bitrate, 2500)
        self.preview = preview_factory(neat, self.args) if not args.no_stream else None
        self.stopping, self.pausing, self.parked = threading.Event(), threading.Event(), threading.Event()
        self.frame = None
        self.frame_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='scout-usb', daemon=True)
        self.thread.start()

    def _run(self):
        import cv2
        neat, args = self.neat, self.args
        cap = runner = model = sender = None
        stamps = deque(maxlen=60)
        sequence = 0
        started = time.monotonic_ns()
        try:
            options = neat.RunOptions()
            options.queue_depth = 1
            options.advanced.copy_input = False
            route = neat.ModelRouteOptions()
            route.upstream_name = route.buffer_name = 'usb0'
            route.name_suffix = '_usb0'
            if not args.no_stream:
                meta = neat.MetadataSenderOptions()
                meta.host, meta.channel = args.host, args.channel
                meta.metadata_port_base = args.metadata_port_base
                sender = neat.MetadataSender(meta)
            while not self.stopping.is_set():
                if self.pausing.is_set():
                    if cap is not None:
                        cap.release()
                        cap = None
                    if runner is not None:
                        runner.close()
                    runner = model = None
                    with self.state.lock:
                        self.state.watch.unavailable('paused')
                    self.parked.set()
                    while self.pausing.is_set() and not self.stopping.wait(.05):
                        pass
                    self.parked.clear()
                    stamps.clear()
                    continue
                tick = time.monotonic()
                if model is None:
                    model = detector.make_model(neat, args, input_format=neat.PreprocessColorFormat.BGR)
                if cap is None:
                    try:
                        cap = cv2.VideoCapture(resolve_camera(args.usb_camera), cv2.CAP_V4L2)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                        if not cap.isOpened():
                            raise RuntimeError('Cannot open USB camera')
                    except Exception as exc:
                        if cap is not None:
                            cap.release()
                        cap = None
                        with self.state.lock:
                            self.state.watch.unavailable('offline', str(exc))
                        self.stopping.wait(1)
                        continue
                ok, bgr = cap.read()
                received_ns = time.monotonic_ns()
                if not ok:
                    cap.release()
                    cap = None
                    with self.state.lock:
                        self.state.watch.unavailable('offline', 'USB capture stopped; reconnect the camera.')
                    with self.frame_lock:
                        self.frame = None
                    self.stopping.wait(.5)
                    continue
                if bgr.shape[:2] != (args.height, args.width):
                    raise RuntimeError(f'USB negotiated {bgr.shape[1]}x{bgr.shape[0]}, expected {args.width}x{args.height}')
                pts_ns = received_ns - started
                sequence += 1
                tensor = neat.Tensor.from_numpy(bgr, copy=True, image_format=neat.PixelFormat.BGR,
                                                memory=neat.TensorMemory.EV74)
                frame = neat.make_tensor_sample('image', tensor)
                frame.pts_ns, frame.frame_id = pts_ns, sequence
                frame.duration_ns = 1_000_000_000 // args.fps
                if self.preview:
                    self.preview.send(frame, pts_ns)
                if runner is None:
                    runner = model.build([frame], route_options=route, run_options=options)
                result = runner.run([frame], timeout_ms=5000)
                objects = detector.detection_objects(neat, result, args, self.state.labels)
                now = time.monotonic()
                stamps.append(now)
                with self.state.lock:
                    watch = self.state.watch
                    transition = watch.observe(objects, now, args.width, args.height)
                    if len(stamps) > 1:
                        watch.fps = (len(stamps)-1) / (stamps[-1]-stamps[0])
                    metadata = watch.snapshot()
                    if transition:
                        self.state.event('zone', 'Area occupied' if transition['occupied'] else 'No watched objects',
                            ', '.join(transition['labels']) if transition['occupied'] else 'No selected object class detected in the marked area.',
                            self.jpeg_image(bgr), camera_id='usb', camera_label='USB 02',
                            source_pts_ms=pts_ns // 1_000_000, captured_time=time.time())
                with self.frame_lock:
                    self.frame = (bgr, pts_ns, time.monotonic(), time.time())
                if sender:
                    sender.send_metadata('object-detection', json.dumps(metadata), pts_ns // 1_000_000, str(sequence))
                result = frame = tensor = None
                self.stopping.wait(max(0, 1/args.fps - (time.monotonic()-tick)))
        except Exception as exc:
            print('USB watch error: ' + str(exc), flush=True)
            with self.state.lock:
                self.state.watch.unavailable('error', str(exc))
        finally:
            if cap is not None:
                cap.release()
            if runner is not None:
                runner.close()
            runner = model = None
            self.parked.set()
            # Keep the encoder until global shutdown, even if USB goes offline.

    def pause(self):
        self.pausing.set()
        if not self.parked.wait(8):
            raise RuntimeError('USB inference did not pause; cannot start Gemma safely')

    def resume(self):
        self.pausing.clear()

    def inspection_job(self):
        with self.state.lock:
            watch = self.state.watch
            if not watch.inspection_requested or watch.status != 'live':
                return None
            generation, zone = watch.generation, list(watch.zone)
            with self.frame_lock:
                if self.frame is None or time.monotonic() - self.frame[2] > 2:
                    return None
                bgr, pts_ns, captured_at, captured_time = self.frame
            h, w = bgr.shape[:2]
            x, y, zw, zh = zone
            jpeg = self.jpeg_image(bgr[int(y*h):int((y+zh)*h), int(x*w):int((x+zw)*w)])
            watch.inspection_requested, watch.checking = False, True
            return dict(camera_id='usb', generation=generation, jpeg=jpeg, pts_ms=pts_ns//1_000_000,
                        captured_at=captured_at, captured_time=captured_time, object_class=watch.object_class,
                        query='Inspect the landing/payload watch area')

    def apply_result(self, job, result):
        with self.state.lock:
            watch = self.state.watch
            watch.checking = False
            self.state.busy = False
            self.state.vlm_latency = result.get('latency_s')
            if job['generation'] != watch.generation:
                self.state.stale_results += 1
                return
            verdict = result.get('verdict')
            watch.last_verdict = verdict
            title = ('Area check inconclusive' if not verdict else
                     {'yes': 'Objects visible in area', 'no': 'No watched objects visible',
                      'uncertain': 'Area needs a clearer view'}[verdict['match']])
            self.state.event('check', title, verdict['reason'] if verdict else 'Try another clear view.',
                job['jpeg'], camera_id='usb', camera_label='USB 02',
                source_pts_ms=job['pts_ms'], captured_time=job['captured_time'],
                latency_s=result.get('latency_s'), verdict=verdict['match'] if verdict else 'uncertain')

    def close(self):
        self.stopping.set()
        self.pausing.clear()
        self.thread.join(10)
        if self.thread.is_alive():
            raise RuntimeError('USB worker did not stop')
        if self.preview:
            self.preview.close()
        with self.state.lock:
            if self.state.watch.status != 'error':
                self.state.watch.unavailable('stopped')
