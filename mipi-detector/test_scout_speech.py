"""Configuration and TTS ownership checks; no speech model is loaded."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scout_config import DEFAULT_SPEECH, configure_defaults, load_config, merge_settings
from scout_speech import Speech
from scout_lifecycle import shutdown_all


class ConfigurationTests(unittest.TestCase):
    def test_legacy_launch_does_not_require_yaml_or_speech(self):
        defaults, config = load_config([])
        self.assertEqual(defaults, {})
        self.assertFalse(config['speech']['stt']['enabled'])
        self.assertFalse(config['speech']['tts']['enabled'])

    def test_rejects_misspelled_and_wrongly_typed_settings(self):
        for settings in ({'tts': {'enabled': 'false'}}, {'enable': True},
                         {'tts': {'speed': float('nan')}}, {'stt': {'record_seconds': 1.2}}):
            with self.assertRaises(ValueError):
                merge_settings(DEFAULT_SPEECH, settings)

    def test_cli_overrides_yaml_defaults(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--fps', type=int, default=25)
        parser.add_argument('--model', type=Path)
        configure_defaults(parser, {'fps': 30, 'model': '/models/example'})
        self.assertEqual(parser.parse_args(['--fps', '15']).fps, 15)
        self.assertEqual(parser.parse_args([]).model, Path('/models/example'))
        with self.assertRaises(ValueError):
            configure_defaults(parser, {'typo': True})

    def test_periodic_inspection_boolean_yaml_and_cli_overrides(self):
        import argparse
        from scout import configure
        parser = argparse.ArgumentParser()
        configure(parser)
        self.assertTrue(parser.parse_args([]).auto_checks)
        self.assertEqual(parser.parse_args([]).inspection_interval, 3)
        configure_defaults(parser, {'auto_checks': False, 'inspection_interval': 4})
        self.assertFalse(parser.parse_args([]).auto_checks)
        self.assertTrue(parser.parse_args(['--auto-checks']).auto_checks)
        self.assertFalse(parser.parse_args(['--no-auto-checks']).auto_checks)
        with self.assertRaises(ValueError):
            configure_defaults(parser, {'auto_checks': 'false'})


class SpeechReplyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        settings = deepcopy(DEFAULT_SPEECH)
        settings['enabled'] = settings['tts']['enabled'] = True
        settings['runtime_dir'] = self.temporary.name
        self.speech = Speech({'speech': settings, 'model_servers': {}})
        # Request tests use a ready fake service; they never load a speech model.
        self.speech.models_ready = True
        self.owner = 'browser_session_1234'
        self.event = dict(id='42', title='Snapshot matches', detail='A blue backpack is visible.', camera_id='mipi')

    def tearDown(self):
        self.speech.close()
        self.temporary.cleanup()

    def queue(self):
        self.assertTrue(self.speech.speak(self.owner, self.event, 2))

    def test_only_one_pending_request_and_no_duplicate_evidence(self):
        self.queue()
        self.assertFalse(self.speech.speak(self.owner, {**self.event, 'id': '43'}, 2))
        self.speech.cancel()
        self.assertFalse(self.speech.speak(self.owner, self.event, 2))
        self.assertTrue(self.speech.speak(self.owner, {**self.event, 'id': '43'}, 2))

    def test_disabled_tts_is_a_noop(self):
        self.speech.settings['tts']['enabled'] = False
        self.assertFalse(self.speech.speak(self.owner, self.event, 2))
        self.assertIsNone(self.speech.job)

    def test_unrelated_camera_and_session_do_not_cancel_reply(self):
        self.queue()
        self.speech.cancel(owner='another_browser_123')
        self.speech.cancel(camera='usb')
        self.assertEqual(self.speech.phase, 'queued')
        self.speech.cancel(camera='mipi')
        self.assertEqual(self.speech.phase, 'cancelled')

    def test_audio_requires_current_token_and_expires_on_reset(self):
        self.queue()
        self.speech._make_directory()
        self.speech.job['audio'] = ['audio-0.wav']
        self.speech.phase = 'ready'
        (self.speech.directory / 'audio-0.wav').write_bytes(b'wave')
        token = self.speech.job['id']
        self.assertNotIn('audio', self.speech.snapshot('other_browser'))
        self.assertEqual(len(self.speech.snapshot(self.owner)['audio']), 1)
        self.assertIsNone(self.speech.audio('wrong', 'audio-0.wav'))
        self.assertIsNone(self.speech.audio(token, '../../request.json'))
        self.assertEqual(self.speech.audio(token, 'audio-0.wav'), b'wave')
        self.speech.cancel(owner=self.owner)
        self.assertIsNone(self.speech.audio(token, 'audio-0.wav'))
        self.assertFalse(list(Path(self.temporary.name).iterdir()))

    def test_cancel_waits_for_native_worker_and_discards_its_audio(self):
        self.queue()
        self.speech._make_directory()
        worker = Mock(pid=987654321)
        worker.poll.return_value = None
        self.speech.process = worker
        self.speech.deadline = float('inf')
        self.speech.cancel()
        self.assertTrue(self.speech.busy)
        self.assertIsNone(self.speech.poll())
        worker.terminate.assert_not_called()
        (self.speech.directory / 'result.json').write_text(json.dumps({'audio': ['audio-0.wav']}))
        worker.poll.return_value = 0
        with patch('scout_speech.os.killpg'):
            result = self.speech.poll()
        self.assertTrue(result['cancelled'])
        self.assertFalse(self.speech.busy)
        self.assertEqual(self.speech.phase, 'cancelled')
        self.assertIsNone(self.speech.directory)

    def test_failed_launcher_releases_slot_and_reports_error(self):
        self.queue()
        with patch('scout_speech.subprocess.Popen', side_effect=FileNotFoundError('Missing launcher')):
            self.assertFalse(self.speech.start_pending())
        self.assertFalse(self.speech.busy)
        self.assertEqual(self.speech.phase, 'error')
        self.assertIn('Missing launcher', self.speech.snapshot(self.owner)['error'])

    def test_request_completion_keeps_model_server_resident(self):
        self.queue()
        self.speech._make_directory()
        worker = Mock(pid=987654321)
        worker.poll.return_value = 0
        self.speech.process = worker
        self.speech.deadline = float('inf')
        server = Mock()
        server.poll.return_value = None
        self.speech.services['supertonic'] = dict(process=server, ready=True, config={})
        (self.speech.directory / 'result.json').write_text(json.dumps({'audio': ['audio-0.wav']}))
        try:
            with patch('scout_speech.os.killpg'):
                result = self.speech.poll()
            self.assertEqual(result['audio'], ['audio-0.wav'])
            server.send_signal.assert_not_called()
            server.terminate.assert_not_called()
            server.wait.assert_not_called()
            self.assertTrue(self.speech.poll_models())
        finally:
            self.speech.services.clear()

    def test_model_exit_is_fatal_even_when_no_request_is_active(self):
        server = Mock()
        server.poll.return_value = 1
        self.speech.services['supertonic'] = dict(process=server, ready=True, config={})
        try:
            with self.assertRaisesRegex(RuntimeError, 'stopping all models'):
                self.speech.poll_models()
            self.assertFalse(self.speech.models_ready)
        finally:
            self.speech.services.clear()

    def test_request_deadline_requires_group_shutdown(self):
        self.queue()
        worker = Mock()
        self.speech.process = worker
        self.speech.deadline = 0
        try:
            with self.assertRaisesRegex(RuntimeError, 'stopping all models'):
                self.speech.poll()
            worker.terminate.assert_not_called()
        finally:
            self.speech.process = None


class GroupShutdownTests(unittest.TestCase):
    def test_all_owners_signalled_before_waiting_and_failure_does_not_skip_any(self):
        events = []
        def close_camera():
            events.append('close camera')
            raise RuntimeError('camera close failed')
        resources = [('camera', lambda: events.append('stop camera'), close_camera),
                     ('Gemma', lambda: events.append('stop Gemma'), lambda: events.append('close Gemma')),
                     ('speech', lambda: events.append('stop speech'), lambda: events.append('close speech'))]
        errors = shutdown_all(resources)
        self.assertEqual(events, ['stop camera', 'stop Gemma', 'stop speech',
                                  'close camera', 'close Gemma', 'close speech'])
        self.assertEqual(errors, ['camera close: camera close failed'])

    def test_signal_failure_still_runs_every_close(self):
        def stop():
            raise RuntimeError('already gone')
        close = Mock()
        other = Mock()
        errors = shutdown_all([('one', stop, close), ('two', other, other)])
        close.assert_called_once()
        self.assertEqual(other.call_count, 2)
        self.assertEqual(errors, ['one stop: already gone'])


class LauncherTests(unittest.TestCase):
    def test_explicit_restart_waits_for_exit_and_uses_a_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / 'python'
            record = root / 'pids'
            fake_python.write_text(f'''#!{sys.executable}
import os
from pathlib import Path
p = Path({str(record)!r})
previous = p.read_text() if p.exists() else ''
if previous:
    assert not Path('/proc/' + previous.strip()).exists(), 'Old process still alive'
with p.open('a') as out:
    out.write(str(os.getpid()) + '\\n')
raise SystemExit(0 if previous else 75)
''')
            fake_python.chmod(0o755)
            result = subprocess.run(['bash', str(Path(__file__).with_name('scout.sh'))],
                                    env={**os.environ, 'PYTHON': str(fake_python)}, timeout=10,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(set(record.read_text().splitlines())), 2)

    def test_unexpected_failure_does_not_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / 'python'
            fake_python.write_text('#!/usr/bin/env bash\nexit 1\n')
            fake_python.chmod(0o755)
            result = subprocess.run(['bash', str(Path(__file__).with_name('scout.sh'))],
                                    env={**os.environ, 'PYTHON': str(fake_python)}, timeout=10)
            self.assertEqual(result.returncode, 1)


if __name__ == '__main__':
    unittest.main()
