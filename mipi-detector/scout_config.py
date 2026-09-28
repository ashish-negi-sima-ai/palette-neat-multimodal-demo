"""Optional YAML settings. CLI arguments take precedence over YAML defaults."""

import argparse
from copy import deepcopy
import math
from pathlib import Path


DEFAULT_SPEECH = {
    'enabled': False,
    'implementation_repo': '/workspace/GitHub/sima-ai',
    'launcher_python': '/workspace/GitHub/sima-ai/jarvic_venv/bin/python',
    'runtime_dir': str(Path.home() / '.cache/neat-scout/speech'),
    'stt': dict(enabled=False, input='browser', activation='toggle_button',
                submit_on_stop=True, target='subject_mission', task='transcribe',
                language='auto', record_seconds=30, sample_rate_hz=16000,
                channels=1, sample_format='pcm_s16le', upload_max_bytes=1048576,
                request_timeout_seconds=120),
    'tts': dict(enabled=False, output='browser', voice='M1', language='en', speed=1.,
                speak_subject_results=True, speak_watch_inspection_results=True,
                speak_occupancy_events=False, request_timeout_seconds=120),
    'scheduler': dict(serialize_inference=False, pause_capture_for=[],
                      speech_worker_lifetime='application', release_gemma_for_speech=False,
                      keep_preview_encoders=True, max_pending_voice_jobs=1,
                      stop_all_models_together=True,
                      startup_timeout_seconds=180),
}


def merge_settings(defaults, supplied, prefix='speech'):
    if not isinstance(supplied, dict):
        raise ValueError(f'{prefix} must be a mapping')
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if key not in defaults:
            raise ValueError(f'Unknown setting: {prefix}.{key}')
        expected = defaults[key]
        if isinstance(expected, dict):
            result[key] = merge_settings(expected, value, f'{prefix}.{key}')
        else:
            if isinstance(expected, bool):
                valid = type(value) is bool
            elif isinstance(expected, (int, float)):
                valid = type(value) in (int, float) and math.isfinite(value)
                if isinstance(expected, int):
                    valid = valid and type(value) is int
            else:
                valid = isinstance(value, type(expected))
            if not valid:
                raise ValueError(f'Invalid type for {prefix}.{key}')
            result[key] = value
    return result


def load_config(argv):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    probe.add_argument('--no-stt', action='store_true')
    probe.add_argument('--no-tts', action='store_true')
    options, _ = probe.parse_known_args(argv)
    raw = {}
    if options.config:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML configuration requires PyYAML in the SCOUT Python environment') from exc
        raw = yaml.safe_load(options.config.read_text())
        if not isinstance(raw, dict):
            raise ValueError('Configuration must be a YAML mapping')
        if set(raw) - {'version', 'scout', 'speech', 'model_servers'}:
            raise ValueError('Unknown top-level configuration field')
        if type(raw.get('version')) is not int or raw['version'] != 1:
            raise ValueError('Configuration version must be 1')
    speech = merge_settings(DEFAULT_SPEECH, raw.get('speech', {}))
    stt, tts, scheduler = (speech[key] for key in ('stt', 'tts', 'scheduler'))
    for section, keys in ((stt, ('input', 'activation', 'submit_on_stop', 'target',
                                'sample_rate_hz', 'channels', 'sample_format')),
                          (tts, ('output', 'speak_occupancy_events')),
                          (scheduler, tuple(k for k in scheduler if k != 'startup_timeout_seconds'))):
        default = DEFAULT_SPEECH['stt' if section is stt else 'tts' if section is tts else 'scheduler']
        for key in keys:
            if section[key] != default[key]:
                raise ValueError(f'Unsupported speech setting: {key}={section[key]!r}')
    if stt['task'] not in ('transcribe', 'translate'):
        raise ValueError('stt.task must be transcribe or translate')
    for section, key, low, high in ((stt, 'record_seconds', 1, 30),
                                    (stt, 'upload_max_bytes', 32044, 1048576),
                                    (stt, 'request_timeout_seconds', 1, 300),
                                    (tts, 'request_timeout_seconds', 1, 300),
                                    (tts, 'speed', .5, 2),
                                    (scheduler, 'startup_timeout_seconds', 5, 600)):
        if not low <= section[key] <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
    if stt['upload_max_bytes'] < 44 + stt['record_seconds'] * 32000:
        raise ValueError('upload_max_bytes must accommodate record_seconds of PCM WAV')
    if tts['voice'] not in {f'{sex}{n}' for sex in ('M', 'F') for n in range(1, 6)}:
        raise ValueError('tts.voice must be M1–M5 or F1–F5')
    if tts['language'] not in ('en', 'ko', 'es', 'pt', 'fr'):
        raise ValueError('Unsupported Supertonic language')
    stt['enabled'] = speech['enabled'] and stt['enabled'] and not options.no_stt
    tts['enabled'] = speech['enabled'] and tts['enabled'] and not options.no_tts
    servers = raw.get('model_servers', {})
    if not isinstance(servers, dict) or set(servers) - {'whisper', 'supertonic'}:
        raise ValueError('model_servers may contain only whisper and supertonic')
    for kind, service in (('stt', 'whisper'), ('tts', 'supertonic')):
        if not speech[kind]['enabled']:
            continue
        if not isinstance(servers.get(service), dict):
            raise ValueError(f'model_servers.{service} is required')
        server = servers[service]
        if server.get('host') != '127.0.0.1':
            raise ValueError('Speech model servers must bind to 127.0.0.1')
        if type(server.get('port')) is not int or not 1024 <= server['port'] <= 65535:
            raise ValueError(f'Invalid {service} port')
    if len(servers) == 2 and servers['whisper'].get('port') == servers['supertonic'].get('port'):
        raise ValueError('Speech model servers need different ports')
    return raw.get('scout', {}), dict(speech=speech, model_servers=servers)


def configure_defaults(parser, values):
    if not isinstance(values, dict):
        raise ValueError('scout must be a mapping')
    actions = {a.dest: a for a in parser._actions if a.dest not in ('help', 'config', 'no_stt', 'no_tts')}
    defaults = {}
    for key, value in values.items():
        if key not in actions:
            raise ValueError(f'Unknown scout setting: {key}')
        action = actions[key]
        if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
            if type(value) is not bool:
                raise ValueError(f'scout.{key} must be a boolean')
        elif action.type:
            if isinstance(value, bool):
                raise ValueError(f'Invalid scout.{key}')
            value = action.type(value)
        elif value is not None and not isinstance(value, str):
            raise ValueError(f'scout.{key} must be a string')
        defaults[key] = value
    parser.set_defaults(**defaults)
