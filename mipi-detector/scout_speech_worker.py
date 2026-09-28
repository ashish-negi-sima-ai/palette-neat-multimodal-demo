#!/usr/bin/env python3
"""One HTTP speech request to SCOUT's resident services; this process owns no models."""

import asyncio
import json
from pathlib import Path
import signal
import sys


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


async def request(directory, config, job):
    repo = Path(config['speech']['implementation_repo'])
    sys.path[:0] = [str(repo), str(repo / 'src')]
    import httpx
    from jarvic.audio.clients import WhisperClient, SupertonicClient

    kind = job['kind']
    name = 'whisper' if kind == 'stt' else 'supertonic'
    server = config['model_servers'][name]
    url = f'http://{server["host"]}:{server["port"]}'
    options = config['speech'][kind]
    timeout = options['request_timeout_seconds']
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http:
        async def infer():
            if kind == 'stt':
                client = WhisperClient(http, url, server['served_name'], options['language'], task=options['task'])
                transcript = await client.transcribe((directory / 'recording.wav').read_bytes())
                return {'transcript': transcript}
            client = SupertonicClient(http, url, voice=options['voice'], language=options['language'], speed=options['speed'])
            chunks, total = [], 0
            async for audio in client.synthesize(job['text']):
                total += len(audio)
                if total > 20 * 1024 * 1024 or len(chunks) >= 32:
                    raise RuntimeError('Synthesized reply exceeds the audio limit')
                filename = f'audio-{len(chunks)}.wav'
                (directory / filename).write_bytes(audio)
                chunks.append(filename)
            if not chunks:
                raise RuntimeError('Supertonic returned no audio')
            return {'audio': chunks}
        return await asyncio.wait_for(infer(), timeout=timeout)


def main():
    directory = Path(sys.argv[1])
    payload = json.loads((directory / 'request.json').read_text())
    def interrupted(*_):
        raise KeyboardInterrupt('Speech worker stopped')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        result = asyncio.run(request(directory, payload['config'], payload['job']))
    except (Exception, KeyboardInterrupt) as exc:
        # A transport timeout may leave native inference active. Recover as a group.
        cause = exc
        while cause.__cause__ is not None:
            cause = cause.__cause__
        fatal = isinstance(exc, TimeoutError) or 'Timeout' in type(cause).__name__
        result = {'error': str(exc) or 'Speech request interrupted or timed out', 'fatal': fatal}
    write_json(directory / 'result.json', result)


if __name__ == '__main__':
    main()
