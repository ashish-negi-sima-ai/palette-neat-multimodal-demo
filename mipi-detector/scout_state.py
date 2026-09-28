"""Tracking and mission state. No accelerator or web dependencies."""

from collections import deque
from dataclasses import dataclass, field
import json
import math
import threading
import time


def overlap(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - x)
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - y)
    return w * h / max(1, a[2] * a[3] + b[2] * b[3] - w * h)


@dataclass
class Track:
    id: str
    label: str
    bbox: list
    confidence: float
    seen: float
    hits: int = 1
    velocity: tuple = (0.0, 0.0)
    trail: deque = field(default_factory=lambda: deque(maxlen=24))


class Tracker:
    """Conservative, class-aware image-space tracker; expired IDs are never reused."""

    def __init__(self, max_gap=0.8, prefix='T'):
        self.tracks = {}
        self.next_id = 1
        self.max_gap = max_gap
        self.prefix = prefix

    def update(self, objects, now):
        self.tracks = {k: t for k, t in self.tracks.items() if now - t.seen <= self.max_gap}
        pairs = []
        for key, track in self.tracks.items():
            dt = now - track.seen
            predicted = [track.bbox[0] + track.velocity[0] * dt,
                         track.bbox[1] + track.velocity[1] * dt, *track.bbox[2:]]
            for i, obj in enumerate(objects):
                if obj['label'] != track.label:
                    continue
                b = obj['bbox']
                iou = overlap(predicted, b)
                distance = math.hypot(predicted[0] + predicted[2] / 2 - b[0] - b[2] / 2,
                                      predicted[1] + predicted[3] / 2 - b[1] - b[3] / 2)
                scale = max(1, math.hypot(*predicted[2:]))
                ratio = b[2] * b[3] / max(1, predicted[2] * predicted[3])
                if iou >= 0.18 or (distance / scale < 0.25 and 0.5 < ratio < 2):
                    pairs.append((iou - distance / scale * 0.2, key, i))
        assigned_tracks, assigned_objects = set(), set()
        current = []
        for _, key, i in sorted(pairs, reverse=True):
            if key in assigned_tracks or i in assigned_objects:
                continue
            track, obj = self.tracks[key], objects[i]
            dt = max(0.001, now - track.seen)
            track.velocity = tuple(0.5 * track.velocity[j] +
                                   0.5 * (obj['bbox'][j] - track.bbox[j]) / dt for j in (0, 1))
            track.bbox = list(obj['bbox'])
            track.confidence, track.seen = obj['confidence'], now
            track.hits += 1
            current.append(track)
            assigned_tracks.add(key)
            assigned_objects.add(i)
        for i, obj in enumerate(objects):
            if i not in assigned_objects:
                key = f'{self.prefix}{self.next_id:03d}'
                self.next_id += 1
                track = Track(key, obj['label'], list(obj['bbox']), obj['confidence'], now)
                self.tracks[key] = track
                current.append(track)
        for track in current:
            x, y, w, h = track.bbox
            if not track.trail or now - track.trail[-1][2] >= 0.12:
                track.trail.append((x + w / 2, y + h, now))
        return current


def parse_verdict(text):
    """Malformed, truncated and free-form answers cannot select a subject."""
    result = json.loads(text.strip())
    if not isinstance(result, dict) or set(result) != {'match', 'reason'}:
        raise ValueError('Gemma must return exactly match and reason')
    if result['match'] not in ('yes', 'no', 'uncertain'):
        raise ValueError('Invalid match verdict')
    if not isinstance(result['reason'], str) or not 1 <= len(result['reason'].strip()) <= 240:
        raise ValueError('Invalid evidence explanation')
    return result


class MissionState:
    def __init__(self, labels, auto_checks=True, inspection_interval=3.0):
        self.lock = threading.RLock()
        self.labels = labels
        self.tracker = Tracker()
        self.generation = 0
        self.query = ''
        self.object_class = 'person'
        self.phase = 'idle'
        self.candidate = None
        self.events = deque(maxlen=24)
        self.evidence = {}
        self.event_counter = 0
        self.observed = []
        self.last_frame_at = 0.0
        self.last_verification = None
        self.vlm_status = 'loading'
        self.vlm_error = None
        self.vlm_latency = None
        self.vlm_requests = 0
        self.stale_results = 0
        self.camera_status = 'starting'
        self.camera_error = None
        self.fps = 0.0
        self.frames = 0
        self.busy = False
        self.auto_checks = auto_checks
        self.inspection_interval = inspection_interval
        self.inspection_requested = False
        self.last_inspection_started = -float('inf')
        self.last_inspection = -100.0
        self.last_verdict = None
        self.focus = None
        self.checked_at = {}
        self.track_verdicts = {}
        self.zone_enabled = False
        self.zone = [0.65, 0.48, 0.30, 0.46]
        self.zone_inside = False
        self.watch = None

    def event(self, kind, title, detail='', jpeg=None, **extra):
        extra.setdefault('camera_id', 'mipi')
        extra.setdefault('camera_label', 'MIPI 01')
        self.event_counter += 1
        key = str(self.event_counter)
        event = dict(id=key, kind=kind, title=title, detail=detail,
                     time=time.time(), **extra)
        if jpeg:
            self.evidence[key] = jpeg
            event['image'] = f'/api/evidence/{key}.jpg'
        self.events.appendleft(event)
        keep = {e['id'] for e in self.events}
        self.evidence = {k: v for k, v in self.evidence.items() if k in keep}
        return event

    def start(self, query, object_class, zone=False):
        if not isinstance(query, str) or not 3 <= len(query.strip()) <= 180:
            raise ValueError('Describe the subject in 3–180 characters.')
        if object_class not in self.labels:
            raise ValueError('Choose a supported object class.')
        if not isinstance(zone, bool):
            raise ValueError('Zone watch must be a boolean.')
        with self.lock:
            changed = query.strip() != self.query or object_class != self.object_class
            self.generation += 1
            self.query, self.object_class = query.strip(), object_class
            self.phase = 'searching'
            self.candidate = None
            self.last_verification = None
            self.last_verdict = None
            self.focus = None
            self.checked_at.clear()
            self.track_verdicts.clear()
            self.last_inspection_started = -float('inf')
            self.inspection_requested = True
            self.zone_enabled, self.zone_inside = zone, False
            if changed:
                self.events = deque((e for e in self.events if e.get('camera_id') == 'usb'), maxlen=24)
                self.event('mission', 'Mission started', self.query)

    def reset(self):
        with self.lock:
            self.generation += 1
            self.query = ''
            self.phase = 'idle'
            self.candidate = None
            self.zone_enabled = False
            self.events = deque((e for e in self.events if e.get('camera_id') == 'usb'), maxlen=24)
            keep = {e['id'] for e in self.events}
            self.evidence = {k: v for k, v in self.evidence.items() if k in keep}
            self.last_verification = None
            self.last_verdict = None
            self.focus = None
            self.checked_at.clear()
            self.track_verdicts.clear()
            self.inspection_requested = False

    def pause_capture(self):
        """A stopped camera breaks identity continuity; never bridge that gap."""
        with self.lock:
            self.camera_status = 'paused'
            self.tracker.tracks.clear()
            self.observed.clear()
            self.focus = None
            self.checked_at.clear()
            self.track_verdicts.clear()
            self.zone_inside = False

    def apply_snapshot_result(self, job, result, now):
        """Only a positive snapshot can focus its still-live track in this mission."""
        with self.lock:
            self.busy = False
            self.candidate = None
            self.vlm_latency = result.get('latency_s')
            self.last_inspection = now
            if job['generation'] != self.generation or not self.query:
                self.stale_results += 1
                return False
            if result.get('error'):
                if self.focus and self.focus['track_id'] == job['track_id']:
                    self.focus = None
                self.phase = 'searching' if self.auto_checks else 'reviewed'
                self.event('error', 'Check inconclusive',
                           'The model could not complete this check. Try another clear view.',
                           job['jpeg'], track_id=job['track_id'], source_pts_ms=job['pts_ms'],
                           captured_time=job['captured_time'])
                return False
            verdict = result['verdict']
            self.last_verdict = verdict
            self.last_verification = time.time()
            track_id = job['track_id']
            previous = self.track_verdicts.get(track_id)
            # Never carry confirmation across an expired ID or capture restart.
            track = self.tracker.tracks.get(track_id)
            if track and now - track.seen <= self.tracker.max_gap:
                self.track_verdicts[track_id] = verdict['match']
                if verdict['match'] == 'yes':
                    self.focus = dict(track_id=track_id, verified_at=self.last_verification)
                elif self.focus and self.focus['track_id'] == track_id:
                    self.focus = None
            elif self.focus and self.focus['track_id'] == track_id:
                self.focus = None
            self.phase = 'reviewed' if self.focus or not self.auto_checks else 'searching'
            title = {'yes': 'Snapshot matches', 'no': 'Snapshot does not match',
                     'uncertain': 'Need a clearer view'}[verdict['match']]
            self.event('verified' if verdict['match'] == 'yes' else 'check', title,
                       verdict['reason'], job['jpeg'], track_id=job['track_id'],
                       verdict=verdict['match'], source_pts_ms=job['pts_ms'],
                       latency_s=result['latency_s'], captured_time=job['captured_time'],
                       announce=previous != verdict['match'])
            return verdict['match'] == 'yes'

    def observe(self, objects, now, width, height):
        with self.lock:
            self.observed = self.tracker.update(objects, now)
            active_ids = self.tracker.tracks.keys()
            self.checked_at = {k: v for k, v in self.checked_at.items() if k in active_ids}
            self.track_verdicts = {k: v for k, v in self.track_verdicts.items() if k in active_ids}
            if self.focus and self.focus['track_id'] not in active_ids:
                self.focus = None
                if self.query and self.auto_checks and not self.candidate:
                    self.phase = 'searching'
            self.last_frame_at = now
            self.frames += 1
            self.camera_status = 'live'
            if self.zone_enabled and self.query:
                zx, zy, zw, zh = self.zone
                inside = any(t.label == self.object_class
                             and zx <= (t.bbox[0] + t.bbox[2] / 2) / width <= zx + zw
                             and zy <= (t.bbox[1] + t.bbox[3]) / height <= zy + zh
                             for t in self.observed)
                if inside != self.zone_inside:
                    self.event('zone', 'Object entered watch zone' if inside else 'Watch zone clear',
                               f'{self.object_class} · detector observation')
                self.zone_inside = inside
            return [dict(id=t.id, label=t.label, confidence=t.confidence, bbox=t.bbox,
                         trail=[list(p[:2]) for p in t.trail],
                         checking=t.id == self.candidate)
                    for t in self.observed]

    def choose_candidate(self, now):
        with self.lock:
            if not self.query or self.busy or self.vlm_status != 'ready':
                return None
            if not self.inspection_requested and not (self.auto_checks and
                    now >= self.last_inspection_started + self.inspection_interval):
                return None
            eligible = [t for t in self.observed if t.label == self.object_class and t.hits >= 3]
            if not eligible:
                return None
            if self.focus:
                # Recheck the confirmed identity before considering a replacement.
                return next((t for t in eligible if t.id == self.focus['track_id']), None)
            # Try unseen tracks first, then the least recently inspected track.
            return min(eligible, key=lambda t: (self.checked_at.get(t.id, -float('inf')), -t.confidence))

    def begin_inspection(self, track_id, now):
        """Record the actual capture/start time, without queuing missed intervals."""
        with self.lock:
            self.candidate = track_id
            self.busy = True
            self.inspection_requested = False
            self.last_inspection_started = now
            self.checked_at[track_id] = now
            self.vlm_requests += 1
            self.phase = 'verifying'

    def snapshot(self):
        with self.lock:
            return dict(mission_id=self.generation, query=self.query, object_class=self.object_class,
                        phase=self.phase, candidate=self.candidate,
                        focus=dict(self.focus) if self.focus else None,
                        events=list(self.events), last_verification=self.last_verification,
                        last_verdict=self.last_verdict, auto_checks=self.auto_checks,
                        inspection_interval_s=self.inspection_interval,
                        vlm=dict(status=self.vlm_status, error=self.vlm_error, busy=self.busy,
                                 latency_s=self.vlm_latency, requests=self.vlm_requests,
                                 stale_results=self.stale_results),
                        camera=dict(status=self.camera_status, error=self.camera_error,
                                    fps=round(self.fps, 1), frames=self.frames,
                                    observation_age_s=round(time.monotonic() - self.last_frame_at, 2)
                                    if self.last_frame_at else None),
                        zone=dict(enabled=self.zone_enabled, inside=self.zone_inside, bbox=self.zone),
                        watch=self.watch.snapshot() if self.watch else None)
