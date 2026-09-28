"""Resident speech servers and bounded requests, owned by SCOUT's model group."""

import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import wave
from urllib.request import build_opener, ProxyHandler

from scout_lifecycle import shutdown_all


def client_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,80}', value):
        raise ValueError('Invalid browser session')
    return value


class Speech:
    def __init__(self, config):
        self.config = config
        self.settings = config['speech']
        self.lock = threading.RLock()
        self.job = None
        self.process = None
        self.directory = None
        self.log = None
        self.deadline = 0
        self.last_spoken = None
        self.phase = 'idle'
        self.services = {}
        self.services_directory = None
        self.models_ready = not any(self.settings[k]['enabled'] for k in ('stt', 'tts'))
        self.models_phase = 'disabled' if self.models_ready else 'stopped'
        self.http = build_opener(ProxyHandler({}))

    def start_models(self):
        """Start enabled services once. A stopped service requires a group restart."""
        if self.services:
            raise RuntimeError('Speech models already started; restart the whole model group')
        if not any(self.settings[k]['enabled'] for k in ('stt', 'tts')):
            return
        root = Path(self.settings['runtime_dir']).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.services_directory = Path(tempfile.mkdtemp(prefix='models-', dir=root))
        settings_path = self.services_directory / 'servers.json'
        settings_path.write_text(json.dumps({'version': 1, 'model_servers': self.config['model_servers']}))
        repo = Path(self.settings['implementation_repo'])
        environment = os.environ.copy()
        environment.pop('GST_PLUGIN_SYSTEM_PATH_1_0', None)
        environment['PYTHONPATH'] = os.pathsep.join([str(repo), str(repo / 'src')])
        environment['PYTHONUNBUFFERED'] = '1'
        self.models_ready, self.models_phase = False, 'loading'
        for kind, name in (('stt', 'whisper'), ('tts', 'supertonic')):
            if not self.settings[kind]['enabled']:
                continue
            server = self.config['model_servers'][name]
            import socket
            with socket.socket() as probe:
                if probe.connect_ex((server['host'], server['port'])) == 0:
                    raise RuntimeError(f'{name} port is occupied; SCOUT will not take over another model server')
            log = (self.services_directory / f'{name}.log').open('wb')
            try:
                # sima-ai's launcher execs the model Python, preserving this PID.
                process = subprocess.Popen([self.settings['launcher_python'], '-m',
                    'services.speech.start', name, '--config', str(settings_path)],
                    cwd=repo, env=environment, start_new_session=True, stdout=log, stderr=log)
            except BaseException:
                log.close()
                raise
            self.services[name] = dict(process=process, log=log, ready=False, config=server,
                deadline=time.monotonic() + self.settings['scheduler']['startup_timeout_seconds'])
        print(f'Speech model logs: {self.services_directory}', flush=True)

    def poll_models(self):
        """A lost model is fatal to the whole group, even when no request is active."""
        for name, service in self.services.items():
            process, server = service['process'], service['config']
            if process.poll() is not None:
                self.models_ready, self.models_phase = False, 'error'
                raise RuntimeError(f'{name} model stopped ({process.returncode}); stopping all models. Logs: {self.services_directory}')
            if service['ready']:
                continue
            endpoint = '/v1/models' if name == 'whisper' else '/health'
            try:
                with self.http.open(f'http://{server["host"]}:{server["port"]}{endpoint}', timeout=.2) as response:
                    data = json.load(response)
                service['ready'] = (any(item.get('id') == server['served_name'] for item in data.get('data', []))
                                    if name == 'whisper' else data.get('status') == 'ok')
            except (OSError, ValueError):
                pass
            if not service['ready'] and time.monotonic() > service['deadline']:
                raise RuntimeError(f'{name} startup timed out; stopping all models')
        if self.services and all(s['ready'] for s in self.services.values()):
            self.models_ready, self.models_phase = True, 'ready'
        return self.models_ready

    def public_config(self):
        return dict(stt=self.settings['stt']['enabled'], tts=self.settings['tts']['enabled'],
                    record_seconds=self.settings['stt']['record_seconds'])

    def snapshot(self, owner=None):
        with self.lock:
            result = dict(phase=self.phase, busy=self.busy, models=self.models_phase)
            if self.job and owner == self.job['client_id']:
                result.update({k: v for k, v in self.job.items() if k in
                               ('id', 'kind', 'error', 'transcript', 'cancelled', 'mission_id')})
                if self.phase == 'ready' and not self.job.get('cancelled'):
                    result['audio'] = [f'/api/speech/audio/{self.job["id"]}/{name}'
                                       for name in self.job.get('audio', [])]
            return result

    @property
    def busy(self):
        return self.process is not None or self.phase in ('recording', 'queued')

    def _new(self, kind, owner, generation, camera_id='mipi', **fields):
        if self.busy:
            raise ValueError('A speech request is already in progress')
        self._remove_files()
        self.job = dict(id=secrets.token_urlsafe(24), client_id=client_id(owner), kind=kind,
                        generation=generation, camera_id=camera_id, **fields)
        self.phase = 'queued'
        return self.job

    def begin(self, owner, generation, object_class, zone):
        with self.lock:
            if not self.settings['stt']['enabled']:
                raise ValueError('Speech recognition is disabled in configuration')
            if not self.models_ready:
                raise ValueError('Wait for the speech models to finish loading')
            job = self._new('stt', owner, generation, object_class=object_class, zone=zone)
            self.phase = 'recording'
            # Recover the single slot if a browser disappears during recording.
            self.deadline = time.monotonic() + self.settings['stt']['record_seconds'] + 30
            return job['id']

    def upload(self, token, audio):
        with self.lock:
            if not self.job or token != self.job['id'] or self.phase != 'recording':
                raise ValueError('Recording expired or was cancelled')
            options = self.settings['stt']
            if len(audio) > options['upload_max_bytes']:
                raise ValueError('Recording too large')
            try:
                with wave.open(io.BytesIO(audio), 'rb') as wav:
                    frames = wav.getnframes()
                    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, 'NONE'):
                        raise ValueError('Expected mono 16 kHz PCM16 WAV')
                    if not 0 < frames <= options['record_seconds'] * 16000:
                        raise ValueError('Recording is empty or too long')
                    if len(wav.readframes(frames)) != frames * 2:
                        raise ValueError('Truncated recording')
            except (wave.Error, EOFError) as exc:
                raise ValueError('Invalid WAV recording') from exc
            self._make_directory()
            (self.directory / 'recording.wav').write_bytes(audio)
            self.phase = 'queued'

    def speak(self, owner, event, generation):
        with self.lock:
            camera = event.get('camera_id', 'mipi')
            enabled = self.settings['tts']['speak_watch_inspection_results' if camera == 'usb' else 'speak_subject_results']
            if not owner or not self.settings['tts']['enabled'] or not enabled or self.busy:
                return False
            key = (camera, generation, event['id'])
            if self.last_spoken == key:
                return False
            self._new('tts', owner, generation, camera, text=(event['title'] + '. ' + event['detail'])[:800])
            self.last_spoken = key
            return True

    def cancel(self, owner=None, camera=None):
        with self.lock:
            if not self.job or (owner is not None and owner != self.job['client_id']):
                return
            if camera is not None and camera != self.job['camera_id']:
                return
            self.job['cancelled'] = True
            # Invalidate this request; cancellation must not unload a resident model.
            self.phase = 'cancelling' if self.process else 'cancelled'
            if not self.process:
                self._remove_files()

    def _make_directory(self):
        if self.directory is None:
            root = Path(self.settings['runtime_dir']).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix='job-', dir=root))

    def _remove_files(self):
        if self.directory:
            shutil.rmtree(self.directory)
            self.directory = None

    def start_pending(self):
        """Start an HTTP client only; the model servers remain resident."""
        with self.lock:
            if self.phase != 'queued' or self.process or not self.models_ready:
                return False
            try:
                self._make_directory()
                payload = dict(config=self.config, job=self.job)
                (self.directory / 'request.json').write_text(json.dumps(payload))
                self.log = (self.directory / 'worker.log').open('wb')
                self.process = subprocess.Popen([self.settings['launcher_python'], '-u',
                    str(Path(__file__).with_name('scout_speech_worker.py')), str(self.directory)],
                    start_new_session=True, stdout=self.log, stderr=self.log)
            except Exception as exc:
                if self.log:
                    self.log.close()
                self.log = None
                self.job['error'] = str(exc)
                self.phase = 'error'
                return False
            self.phase = 'transcribing' if self.job['kind'] == 'stt' else 'synthesizing'
            self.deadline = time.monotonic() + self.settings[self.job['kind']]['request_timeout_seconds'] + 10
            return True

    def poll(self):
        """Return a completed request, without stopping its model."""
        with self.lock:
            if self.phase == 'recording' and time.monotonic() > self.deadline:
                self.cancel()
            if not self.process:
                return None
            expired = time.monotonic() > self.deadline
            if expired:
                raise RuntimeError('Speech request exceeded its deadline; stopping all models')
            else:
                if self.process.poll() is None:
                    progress = self.directory / 'progress.json'
                    if progress.exists() and not self.job.get('cancelled'):
                        self.phase = json.loads(progress.read_text())['phase']
                    return None
                try:
                    result = json.loads((self.directory / 'result.json').read_text())
                except (OSError, ValueError):
                    raise RuntimeError('Speech client exited without a result; stopping all models')
                self._stop_process()
            if result.get('fatal'):
                raise RuntimeError(result['error'] + '; stopping all models')
            self.job.update(result)
            self.phase = 'cancelled' if self.job.get('cancelled') else 'error' if result.get('error') else 'ready'
            recording = self.directory / 'recording.wav'
            recording.unlink(missing_ok=True)
            if self.job.get('cancelled'):
                self._remove_files()
            return self.job

    def audio(self, token, name):
        with self.lock:
            if (self.phase == 'ready' and self.job and token == self.job['id']
                    and name in self.job.get('audio', []) and self.directory):
                return (self.directory / name).read_bytes()
            return None

    def _stop_process(self):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(14)
                except subprocess.TimeoutExpired:
                    pass
            # This process is only an HTTP client; it owns no model servers.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    def request_stop(self):
        """Called with every other model owner's stop signal during group shutdown."""
        for service in self.services.values():
            if service['process'].poll() is None:
                try:
                    # Supertonic performs graceful engine cleanup on SIGINT.
                    service['process'].send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass

    @staticmethod
    def _close_service(service):
        process = service['process']
        try:
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(3)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            service['log'].close()

    def close(self):
        self.request_stop()
        errors = shutdown_all([('speech request', None, self._stop_process)] +
            [(name, None, lambda service=service: self._close_service(service)) for name, service in self.services.items()])
        self.services.clear()
        self.models_ready, self.models_phase = False, 'stopped'
        with self.lock:
            self._remove_files()
        # Keep service logs for startup/crash diagnosis; they contain no recordings.
        if errors:
            raise RuntimeError('; '.join(errors))
