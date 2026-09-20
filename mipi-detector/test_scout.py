"""Mission invariants, independent of camera and accelerator availability."""

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scout import FrameCache
from scout_state import MissionState, Tracker, parse_verdict
from scout_vlm import validate_model


def person(x=10, label='person'):
    return dict(label=label, bbox=[x, 10, 60, 150], confidence=.9)


class PreviewTimingTests(unittest.TestCase):
    def test_restart_preserves_video_timeline_and_capture_matching(self):
        sent = []
        preview = SimpleNamespace(started_ns=1_000_000_000,
                                  send=lambda frame, pts: sent.append(pts))
        for started_ns, expected in ((2_000_000_000, [1_000_000_000, 1_033_333_333]),
                                     (5_000_000_000, [4_000_000_000, 4_033_333_333])):
            frames = [SimpleNamespace(pts_ns=100_000_000),
                      SimpleNamespace(pts_ns=133_333_333)]
            pending = iter(frames)
            drained = threading.Event()
            release = threading.Event()

            def pull(_name, _timeout):
                sample = next(pending, None)
                if sample is None:
                    drained.set()
                    release.wait(.2)
                return sample

            with patch('scout.time.monotonic_ns', return_value=started_ns):
                cache = FrameCache(SimpleNamespace(pull=pull), preview)
                try:
                    self.assertTrue(drained.wait(2))
                    self.assertIsNone(cache.error)
                    self.assertEqual(sent[-2:], expected)
                    # Metadata uses this same offset, while evidence lookup keeps
                    # the original camera PTS after a capture restart.
                    for frame, pts in zip(frames, expected):
                        self.assertEqual(frame.pts_ns + cache.pts_offset_ns, pts)
                        self.assertIs(cache.find(frame.pts_ns), frame)
                    self.assertEqual([f.pts_ns for f in frames], [100_000_000, 133_333_333])
                finally:
                    cache.stopping.set()
                    release.set()
                    cache.close()
        self.assertEqual(sent, sorted(sent))


class TrackingTests(unittest.TestCase):
    def test_track_follows_motion_and_not_detection_order(self):
        tracker = Tracker()
        first = tracker.update([person(10), person(400)], 1)
        ids = [t.id for t in first]
        second = tracker.update([person(403), person(13)], 1.04)
        self.assertEqual({t.id: t.bbox[0] for t in second}, {ids[0]: 13, ids[1]: 403})

    def test_expired_id_is_not_reused_and_classes_do_not_merge(self):
        tracker = Tracker()
        old = tracker.update([person()], 1)[0].id
        bag = tracker.update([person(label='backpack')], 1.04)[0].id
        self.assertNotEqual(old, bag)
        new = tracker.update([person()], 3)[0].id
        self.assertNotEqual(old, new)


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.state = MissionState(['person', 'backpack'])
        self.state.start('a person wearing a blue shirt', 'person')
        self.state.vlm_status = 'ready'
        for t in (1, 1.04, 1.08):
            self.state.observe([person()], t, 1920, 1080)
        self.track = self.state.observed[0]

    def job(self, when=1.08):
        return dict(generation=self.state.generation, track_id=self.track.id,
                    captured_at=when, captured_time=100, pts_ms=1080, jpeg=b'jpeg')

    def result(self, match='yes'):
        return dict(verdict={'match': match, 'reason': 'A blue shirt is visible.'}, latency_s=.6)

    def test_snapshot_verdict_does_not_verify_a_resumed_track(self):
        job = self.job()
        self.state.pause_capture()
        self.assertTrue(self.state.apply_snapshot_result(job, self.result(), 4))
        self.state.observe([person()], 4.1, 1920, 1080)
        self.assertNotEqual(self.state.observed[0].id, self.track.id)
        self.assertEqual(self.state.phase, 'reviewed')
        self.assertEqual(self.state.events[0]['track_id'], self.track.id)
        self.assertNotIn('target', self.state.snapshot())

    def test_new_mission_rejects_old_inflight_answer(self):
        job = self.job()
        self.state.start('an orange backpack', 'backpack')
        self.assertFalse(self.state.apply_snapshot_result(job, self.result(), 4))
        self.assertIsNone(self.state.last_verdict)
        self.assertEqual(self.state.stale_results, 1)
        self.assertEqual(len(self.state.events), 1)

    def test_no_and_uncertain_are_preserved_as_evidence(self):
        for verdict in ('no', 'uncertain'):
            self.assertFalse(self.state.apply_snapshot_result(self.job(), self.result(verdict), 4))
            self.assertEqual(self.state.last_verdict['match'], verdict)
            self.assertEqual(self.state.events[0]['verdict'], verdict)

    def test_on_demand_does_not_repeat_and_auto_checks_wait(self):
        self.assertIsNotNone(self.state.choose_candidate(1.1))
        self.state.inspection_requested = False
        self.state.last_inspection = 4
        self.assertIsNone(self.state.choose_candidate(30))
        self.state.auto_checks = True
        self.assertIsNone(self.state.choose_candidate(18))
        self.assertIsNotNone(self.state.choose_candidate(19))
        self.state.busy = True
        self.assertIsNone(self.state.choose_candidate(20))

    def test_zone_watch_uses_class_and_bottom_center(self):
        self.state.zone_enabled = True
        self.state.zone = [.2, .2, .5, .5]
        inside = dict(label='person', bbox=[300, 200, 100, 300], confidence=.9)
        self.state.observe([inside], 1.2, 1000, 1000)
        self.assertTrue(self.state.zone_inside)
        self.assertEqual(self.state.events[0]['title'], 'Object entered watch zone')
        self.state.observe([], 1.3, 1000, 1000)
        self.assertFalse(self.state.zone_inside)
        self.assertEqual(self.state.events[0]['title'], 'Watch zone clear')

    def test_failed_verification_is_not_a_positive_verdict(self):
        self.state.apply_snapshot_result(self.job(), {'error': 'native error', 'latency_s': 2}, 4)
        self.assertIsNone(self.state.last_verdict)
        self.assertEqual(self.state.events[0]['kind'], 'error')
        self.assertNotIn('native error', self.state.events[0]['detail'])

    def test_evidence_storage_is_bounded(self):
        for i in range(100):
            self.state.event('check', str(i), jpeg=b'jpeg')
        self.assertEqual(len(self.state.events), 24)
        self.assertEqual(len(self.state.evidence), 24)

    def test_reset_discards_evidence_and_late_results(self):
        job = self.job()
        self.state.reset()
        self.state.apply_snapshot_result(job, self.result(), 4)
        self.assertEqual(self.state.phase, 'idle')
        self.assertFalse(self.state.events)

    def test_invalid_mission_does_not_replace_active_mission(self):
        generation = self.state.generation
        for query, label in [('', 'person'), ('x' * 181, 'person'), ('a hat', 'unknown')]:
            with self.assertRaises(ValueError):
                self.state.start(query, label)
        self.assertEqual(generation, self.state.generation)


class VerificationTests(unittest.TestCase):
    def test_strict_verdict_schema(self):
        valid = dict(match='yes', reason='A blue backpack is visible.')
        self.assertEqual(parse_verdict(json.dumps(valid)), valid)
        for bad in ['yes', '{"match": true,"reason":"yes"}', '```json\n{}\n```',
                    '{"match":"yes","reason":"yes","action":"follow"}',
                    '{"match":"yes","reason":""}', '{"match":"yes"']:
            with self.assertRaises(ValueError):
                parse_verdict(bad)

    def test_incomplete_model_fails_before_native_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'devkit').mkdir()
            (root / 'devkit/vlm_config.json').write_text(json.dumps(dict(
                model_type='vlm-gemma4', vision_model_name='vision', language_model_name='language')))
            with self.assertRaisesRegex(ValueError, 'download incomplete'):
                validate_model(root)


if __name__ == '__main__':
    unittest.main()
