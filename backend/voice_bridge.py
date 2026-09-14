#!/usr/bin/env python3
"""
voice_bridge.py - client for the on-device voice runtime (voice/board_voice_server.py).

    python3 backend/voice_bridge.py --health
    python3 backend/voice_bridge.py --wav testdata/reference/ko_02.wav --language ko
    python3 backend/voice_bridge.py --text "왼쪽 사람만 보여줘"

The interface:

    GET  /health                         {state, ready, models, counters, ...}
    POST /voice_command?language=ko|en|auto
                                         body: 16 kHz mono signed-16-bit-LE WAV
                                         -> {ok, stt_text, processed, timing_ms, error}
    POST /text_command                   body: {"text": "..."}
                                         -> {ok, text, processed, timing_ms}

`processed` is backend/command_processor.py's result: the validated command, whether
the parser or Qwen3 produced it, and the text the UI shows. The browser records the
microphone and uploads the WAV; this module only carries bytes.
"""

import argparse
import json
import sys
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8983

HEALTH_TIMEOUT_S = 4.0
REQUEST_TIMEOUT_S = 120.0
HEALTH_POLL_INTERVAL_S = 3.0
# Loading Whisper-medium holds the DevKit busy for seconds, so one missed reply
# during startup is normal; only call it disconnected after a couple in a row.
FAILURES_BEFORE_DISCONNECTED = 2

CONN_DISCONNECTED = "Voice runtime disconnected"
CONN_LOADING = "Voice runtime loading"
CONN_READY = "Voice AI ready"

LANGUAGES = ("ko", "en", "auto")


class VoiceError(Exception):
    pass


class VoiceServerClient:
    """Thin stdlib HTTP client."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 request_timeout=REQUEST_TIMEOUT_S,
                 health_timeout=HEALTH_TIMEOUT_S):
        self.host = host
        self.port = int(port)
        self.base = "http://%s:%d" % (host, self.port)
        self.request_timeout = request_timeout
        self.health_timeout = health_timeout

    def health(self):
        try:
            with urlrequest.urlopen(self.base + "/health",
                                    timeout=self.health_timeout) as resp:
                return json.load(resp)
        except Exception as exc:                                # noqa: BLE001
            raise VoiceError("GET %s/health -> %s" % (self.base, exc)) from exc

    def _post(self, path, data, content_type):
        request = urlrequest.Request(self.base + path, data=data,
                                     headers={"Content-Type": content_type},
                                     method="POST")
        try:
            with urlrequest.urlopen(request, timeout=self.request_timeout) as resp:
                return json.load(resp), resp.status
        except urlerror.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                return json.loads(raw), exc.code
            except ValueError:
                return {"ok": False, "error": "HTTP %s: %s" % (exc.code, raw[:200])}, exc.code
        except Exception as exc:                                # noqa: BLE001
            raise VoiceError("voice runtime unreachable: %s" % exc) from exc

    def voice_command(self, wav_bytes, language="ko"):
        """ONE Whisper + command request; returns (payload, http_status)."""
        if language not in LANGUAGES:
            language = "ko"
        return self._post("/voice_command?language=%s" % language, wav_bytes, "audio/wav")

    def text_command(self, text):
        """ONE typed command through the same command path; (payload, http_status)."""
        body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        return self._post("/text_command", body, "application/json")


class HealthMonitor:
    """Polls /health on its own thread. Independent of everything else."""

    def __init__(self, client, interval=HEALTH_POLL_INTERVAL_S):
        self.client = client
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._snapshot = {"connection": CONN_DISCONNECTED, "ready": False,
                          "state": None, "detail": None, "checked_at": None,
                          "endpoint": client.base, "models": None, "counters": None,
                          "declares_camera_pause": False}
        self._failures = 0

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="voice-health",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, join_timeout=3.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def _loop(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def _poll_once(self):
        try:
            payload = self.client.health()
        except VoiceError as exc:
            self._failures += 1
            if self._failures >= FAILURES_BEFORE_DISCONNECTED:
                self._set({"connection": CONN_DISCONNECTED, "ready": False,
                           "state": None, "detail": str(exc), "models": None,
                           "counters": None})
            return
        self._failures = 0
        ready = bool(payload.get("ready"))
        counters = {k: payload.get(k) for k in ("whisper_runs", "qwen_runs",
                                                "commands_by_parser", "commands_by_qwen",
                                                "commands_not_understood")}
        self._set({"connection": CONN_READY if ready else CONN_LOADING,
                   "ready": ready, "state": payload.get("state"),
                   "detail": payload.get("error"),
                   "models": payload.get("models"), "counters": counters,
                   "busy": payload.get("busy")})

    def _set(self, values):
        with self._lock:
            snapshot = dict(self._snapshot)
            snapshot.update(values)
            snapshot["checked_at"] = time.time()
            self._snapshot = snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--wav", help="send this WAV as one voice command")
    parser.add_argument("--text", help="send this text as one command")
    parser.add_argument("--language", default="ko", choices=LANGUAGES)
    args = parser.parse_args()

    client = VoiceServerClient(args.host, args.port)
    try:
        if args.health:
            print(json.dumps(client.health(), ensure_ascii=False, indent=2))
        if args.wav:
            with open(args.wav, "rb") as handle:
                payload, status = client.voice_command(handle.read(), args.language)
            print("HTTP %s" % status)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.text:
            payload, status = client.text_command(args.text)
            print("HTTP %s" % status)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    except VoiceError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    if not (args.health or args.wav or args.text):
        parser.error("nothing to do; try --health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
