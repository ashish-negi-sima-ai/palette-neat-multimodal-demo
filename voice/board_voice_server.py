#!/usr/bin/env python3
"""
board_voice_server.py - the ON-DEVICE voice and command runtime of the Palette NEAT SDK Demo.

    ${PYNEAT_ENV}/bin/python3 voice/board_voice_server.py --port 8983
    (normally started and stopped by ./run.sh)

Runs ON THE MODALIX DEVKIT. Both GenAI models execute on the Modalix MLA through the
NEAT GenAI Python runtime (pyneat 0.4.0 / LLiMa):

    Whisper-medium  pyneat.genai.ASRModel   MODEL_ROOT/whisper-medium-a16w8
    Qwen3-0.6B      pyneat.genai.GenAIModel MODEL_ROOT/Qwen3-0.6B-Autoround-a16w4

(paths from config/model_manifest.json; a missing artifact stops the server, nothing
is substituted). Both are loaded and warmed up once and stay resident.

ONE command path for voice and text (backend/command_processor.py):

    POST /voice_command?language=ko|en|auto   body: 16 kHz mono 16-bit WAV
      -> Whisper-medium (MLA) -> confidence guard -> process(text)
    POST /text_command                        body: {"text": "..."}
      -> process(text)

    process(text): deterministic parser; only when it cannot decide, Qwen3-0.6B (MLA)
                   with strict JSON output, validated before anything is executed.

    GET /health          status, models, counters
    GET /debug/infer?model=whisper|qwen|both   one isolated inference (diagnostics)

MLA ARBITRATION (--arbiter / --no-arbiter)

With --arbiter every inference first takes the vision process's MLA lease
(vision-local/src/mla_gate.h) so no YOLO inference is in flight while GenAI runs.
What the vision process does with its cameras during that window is ITS setting
(vision.yaml mla.pause_behaviour / --pause-behaviour), not this server's.

No voice model ever runs on the SDK host.
"""

import argparse
import errno
import json
import os
import socket
import struct
import sys
import threading
import time
import traceback
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(VOICE_DIR)
BACKEND_DIR = os.path.join(APP_DIR, "backend")
SCRIPTS_DIR = os.path.join(APP_DIR, "scripts")
TMP_DIR = os.environ.get("DRONE_DEVKIT_TMP", os.path.join(APP_DIR, "runtime", "tmp"))
for d in (BACKEND_DIR, SCRIPTS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)
try:
    import command_config
    import command_normalizer
    import command_processor
    from check_models import model_root
except ImportError as _exc:                                     # noqa: BLE001
    sys.exit("ERROR: cannot import project modules (%s)" % _exc)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("VOICE_PORT", "8983"))
DEFAULT_ARBITER_PORT = int(os.environ.get("ARBITER_PORT", "8974"))
ARBITER_CONNECT_TIMEOUT_S = 2.0
ARBITER_ACQUIRE_TIMEOUT_S = 15.0

# Longest valid answer: {"action":"detect_only","camera":"both","classes":[4-5 names]}.
QWEN_MAX_NEW_TOKENS = 64
# Refuse to ACT on audio Whisper is not confident about (intelligible Korean
# commands score about -0.1 .. -0.4).
MIN_AVG_LOGPROB = -1.0
MAX_AUDIO_BYTES = 32 * 1024 * 1024
MAX_TEXT_CHARS = 300

# Warm-up: Qwen3 must translate this into exactly this, through the same strict
# validator every real answer goes through.
SELFTEST_TEXT = "왼쪽 카메라에서 사람만 보여줘"
SELFTEST_EXPECT = {"action": "detect_only", "camera": "left", "classes": ["person"]}

STATE_STARTING = "starting"
STATE_LOADING = "loading_models"
STATE_VERIFYING = "verifying_models"
STATE_READY = "ready"
STATE_ERROR = "error"

# What Whisper is asked for. `auto` is sent as the language code "auto"; override
# with WHISPER_AUTO_LANGUAGE if a runtime expects another spelling.
LANGUAGE_MODES = {"ko": "ko", "en": "en",
                  "auto": os.environ.get("WHISPER_AUTO_LANGUAGE", "auto")}


def log(msg):
    print("[board_voice] %s" % msg, flush=True)


def looks_like_wav(data):
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def read_memory_info():
    info = {}
    try:
        for line in open("/proc/meminfo"):
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "CmaTotal", "CmaFree"):
                info[key.lower() + "_mb"] = int(rest.strip().split()[0]) // 1024
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                info["server_rss_mb"] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return info


def write_silence_wav(path, seconds, rate=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = int(rate * seconds)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<%dh" % frames, *([0] * frames)))
    return path


def metrics_dict(result):
    """GenerationResult.metrics -> plain dict (whatever this pyneat exposes)."""
    m = getattr(result, "metrics", None)
    if m is None:
        return None
    out = {}
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            value = getattr(m, name)
        except Exception:                                       # noqa: BLE001
            continue
        if isinstance(value, (int, float, str, bool)):
            out[name] = value
    return out or None


class CommandSpec:
    """This project's command vocabulary, parser and Qwen3 prompt."""

    def __init__(self, config_path, prompt_path):
        self.config = command_config.load(config_path)
        with open(prompt_path, encoding="utf-8") as fh:
            self.system_prompt = fh.read().strip()
        self.prompt_path = prompt_path
        self.config_path = self.config.path
        self.config_sha256 = self.config.sha256
        self.normalizer = command_normalizer.CommandNormalizer(self.config)

    def messages(self, neat, text):
        def msg(role, content):
            m = neat.genai.ChatMessage()
            m.role = role
            m.content = content
            return m
        out = [msg("system", self.system_prompt)]
        for example, obj in self.config.few_shot:
            out.append(msg("user", example))
            out.append(msg("assistant", json.dumps(obj, ensure_ascii=False,
                                                   separators=(",", ":"))))
        out.append(msg("user", text))
        return out


class VisionGate:
    """The vision process's MLA lease, scoped to one TCP connection."""

    def __init__(self, port, enabled):
        self.port = port
        self.enabled = enabled

    class _Lease:
        def __init__(self, gate, reason):
            self.gate, self.reason = gate, reason
            self.sock, self.paused = None, False
            self.acquire_ms, self.state = 0, "disabled"

        def __enter__(self):
            if not self.gate.enabled:
                return self
            t0 = time.time()
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.gate.port),
                                                     timeout=ARBITER_CONNECT_TIMEOUT_S)
            except OSError as exc:
                self.state = "unavailable"
                log("vision lease: no arbiter on 127.0.0.1:%d (%s)" % (self.gate.port, exc))
                return self
            try:
                self.sock.settimeout(ARBITER_ACQUIRE_TIMEOUT_S)
                self.sock.sendall(("ACQUIRE %s\n" % self.reason).encode("ascii"))
                reply = self._readline()
                self.acquire_ms = int((time.time() - t0) * 1000)
                if reply.startswith("PAUSED"):
                    self.paused, self.state = True, "paused"
                    log("vision lease: granted after %d ms (%s)" % (self.acquire_ms, reply.strip()))
                else:
                    self.state = "refused"
                    log("vision lease: refused (%s); running anyway" % reply.strip())
                    self._close()
            except OSError as exc:
                self.state = "error"
                log("vision lease: failed (%s); running anyway" % exc)
                self._close()
            return self

        def __exit__(self, *exc):
            if self.sock is None:
                return False
            try:
                if self.paused:
                    self.sock.sendall(b"RELEASE voice-complete\n")
                    self._readline()
            except OSError:
                pass
            finally:
                self._close()
            return False

        def _readline(self):
            chunks = []
            while True:
                ch = self.sock.recv(1)
                if not ch or ch == b"\n":
                    break
                chunks.append(ch)
            return b"".join(chunks).decode("ascii", "replace")

        def _close(self):
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock, self.paused = None, False

    def hold(self, reason):
        return VisionGate._Lease(self, reason)


class BoardVoicePipeline:
    def __init__(self, spec, whisper_dir, qwen_dir, gate):
        self.spec = spec
        self.whisper_dir, self.qwen_dir = whisper_dir, qwen_dir
        self.gate = gate
        self.neat = None
        self.asr = None
        self.llm = None
        self.state, self.error = STATE_STARTING, None
        self.detail = {}
        self.inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._lease_states = []
        self.counters = {"voice_commands": 0, "text_commands": 0,
                         "commands_by_parser": 0, "commands_by_qwen": 0,
                         "commands_not_understood": 0, "qwen_answers_refused": 0,
                         "commands_refused_low_confidence": 0,
                         "whisper_runs": 0, "qwen_runs": 0,
                         "whisper_loads": 0, "qwen_loads": 0}
        self.last_commands = []

    def _set_state(self, state, error=None):
        with self._state_lock:
            self.state, self.error = state, error
        log("state -> %s%s" % (state, (" (%s)" % error) if error else ""))

    def snapshot(self):
        with self._state_lock:
            state, error = self.state, self.error
        return {
            "service": "palette_neat_demo_voice",
            "backend": "modalix-mla",
            "runs_on_vision_board": True,
            # The vision process decides what its cameras do during a lease; the
            # page must not declare a camera pause on its own.
            "declares_camera_pause": False,
            "host_arch": os.uname().machine,
            "state": state, "ready": state == STATE_READY,
            "busy": self.inference_lock.locked(), "error": error,
            "models": {
                "whisper": {"name": "Whisper-medium", "path": self.whisper_dir,
                            "loaded": self.asr is not None,
                            "model_id": self.detail.get("whisper_id"),
                            "runtime": "pyneat.genai.ASRModel", "device": "Modalix MLA"},
                "qwen": {"name": "Qwen3-0.6B", "path": self.qwen_dir,
                         "loaded": self.llm is not None,
                         "model_id": self.detail.get("qwen_id"),
                         "runtime": "pyneat.genai.GenAIModel", "device": "Modalix MLA"},
            },
            "both_resident": self.asr is not None and self.llm is not None and state == STATE_READY,
            **self.counters,
            "load_timing_ms": self.detail.get("load_timing_ms"),
            "mla_arbitration": {"enabled": self.gate.enabled, "arbiter_port": self.gate.port},
            "languages": sorted(LANGUAGE_MODES),
            "memory": read_memory_info(),
            "last_commands": self.last_commands[-10:],
            "config": {"path": self.spec.config_path, "sha256": self.spec.config_sha256},
        }

    # -- load ------------------------------------------------------------------
    def load(self):
        try:
            import pyneat as neat
            self.neat = neat
            timing = {}
            self._set_state(STATE_LOADING)
            t0 = time.time()
            self.asr = neat.genai.ASRModel(self.whisper_dir)
            self.counters["whisper_loads"] += 1
            self.detail["whisper_id"] = self.asr.model_id()
            timing["whisper_open_ms"] = int((time.time() - t0) * 1000)
            t0 = time.time()
            self.llm = neat.genai.GenAIModel(self.qwen_dir)
            self.counters["qwen_loads"] += 1
            self.detail["qwen_id"] = self.llm.model_id()
            timing["qwen_open_ms"] = int((time.time() - t0) * 1000)
            log("Whisper-medium %s and Qwen3-0.6B %s open on the Modalix MLA"
                % (self.detail["whisper_id"], self.detail["qwen_id"]))

            self._set_state(STATE_VERIFYING)
            warm = write_silence_wav(os.path.join(TMP_DIR, "board_voice_warmup.wav"), 0.5)
            t0 = time.time()
            with self.gate.hold("voice_warmup_whisper"):
                self._run_whisper(warm, "en")
            timing["whisper_warmup_ms"] = int((time.time() - t0) * 1000)
            log("warm-up 1/3: Whisper-medium (MLA) on 0.5 s of silence: %d ms"
                % timing["whisper_warmup_ms"])
            t0 = time.time()
            with self.gate.hold("voice_warmup_qwen"):
                raw, _metrics = self._run_qwen(SELFTEST_TEXT)
            timing["qwen_warmup_ms"] = int((time.time() - t0) * 1000)
            obj, why = command_processor.strict_json_object(raw)
            command = None
            if obj is not None:
                evidence = self.spec.normalizer.evidence(SELFTEST_TEXT)
                command, why = command_processor.validate_model_command(
                    obj, evidence, self.spec.config)
            if command != SELFTEST_EXPECT:
                raise RuntimeError("Qwen3 self-test gave %r for %r (%s)"
                                   % (raw, SELFTEST_TEXT, why))
            log("warm-up 2/3: Qwen3-0.6B (MLA) structured self-test %s: %d ms"
                % (json.dumps(command, ensure_ascii=False), timing["qwen_warmup_ms"]))
            t0 = time.time()
            with self.gate.hold("voice_warmup_recheck"):
                self._run_whisper(warm, "en")
            timing["whisper_recheck_ms"] = int((time.time() - t0) * 1000)
            log("warm-up 3/3: Whisper still resident after Qwen: %d ms" % timing["whisper_recheck_ms"])
            self.detail["load_timing_ms"] = timing
            self._set_state(STATE_READY)
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            self._set_state(STATE_ERROR, "%s: %s" % (type(exc).__name__, exc))

    # -- inference ---------------------------------------------------------------
    def _run_whisper(self, wav_path, language):
        req = self.neat.genai.GenerationRequest()
        req.audio_file = wav_path
        req.language = language
        self.counters["whisper_runs"] += 1
        return self.asr.run(req)

    def _run_qwen(self, text):
        # pyneat 0.4.0's GenerationRequest has no temperature / sampling fields,
        # so the runtime's default decoding is used; the strict validator is what
        # makes the output safe to act on.
        req = self.neat.genai.GenerationRequest()
        req.messages = self.spec.messages(self.neat, text)
        req.max_new_tokens = QWEN_MAX_NEW_TOKENS
        req.enable_thinking = False
        self.counters["qwen_runs"] += 1
        result = self.llm.run(req)
        return result.text, metrics_dict(result)

    def _qwen_with_lease(self, text):
        with self.gate.hold("voice_qwen") as lease:
            raw, metrics = self._run_qwen(text)
        self._lease_states.append(lease.state)
        log("Qwen3-0.6B (MLA) raw output: %r" % (raw or "")[:200])
        return raw, metrics

    def process_text(self, text):
        """The shared command path (parser first, Qwen3 fallback). Lock held."""
        result = command_processor.process(text, self.spec.normalizer, self._qwen_with_lease)
        if result["source"] == "parser":
            self.counters["commands_by_parser"] += 1
        elif result["source"] == "qwen3":
            self.counters["commands_by_qwen"] += 1
        else:
            self.counters["commands_not_understood"] += 1
            if result["qwen_invoked"]:
                self.counters["qwen_answers_refused"] += 1
        log("command %r -> %s  [%s%s]" % (
            text, result["display"], result["source"] or "not understood",
            ("; " + result["reason"]) if result["reason"] else ""))
        return result

    def handle_voice_command(self, wav_bytes, language):
        t_start = time.time()
        self._lease_states = []
        timing = {}
        wav_path = os.path.join(TMP_DIR, "board_voice_request.wav")
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(wav_path, "wb") as fh:
            fh.write(wav_bytes)
        try:
            t0 = time.time()
            with self.gate.hold("voice_whisper") as lease:
                result = self._run_whisper(wav_path, LANGUAGE_MODES[language])
            self._lease_states.append(lease.state)
            timing["whisper"] = int((time.time() - t0) * 1000)
            timing["vision_lease_ms"] = lease.acquire_ms
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "stt_text": "", "processed": None, "timing_ms": None,
                    "error": "Whisper inference failed: %s" % exc}
        stt_text = (result.text or "").strip()
        confidence = result.avg_logprob
        self.counters["voice_commands"] += 1
        log("Whisper-medium (MLA, language=%s): %d ms, avg_logprob %s, detected %s -> %r"
            % (language, timing["whisper"], confidence,
               getattr(result, "language", None), stt_text))
        base = {"stt_text": stt_text, "language": language,
                "detected_language": getattr(result, "language", None),
                "whisper_confidence": None if confidence is None else round(confidence, 3),
                "whisper_device": "Modalix MLA", "qwen_device": "Modalix MLA"}
        if not stt_text:
            timing["server_total"] = int((time.time() - t_start) * 1000)
            return dict(base, ok=False, processed=None, timing_ms=timing,
                        error="Whisper produced no text")
        if confidence is not None and confidence < MIN_AVG_LOGPROB:
            self.counters["commands_refused_low_confidence"] += 1
            timing["server_total"] = int((time.time() - t_start) * 1000)
            out = dict(base, ok=False, processed=None, timing_ms=timing, error=None,
                       refused="speech not clear enough (avg_logprob %.2f below %.2f)"
                       % (confidence, MIN_AVG_LOGPROB))
            self._remember(out)
            return out
        processed = self.process_text(stt_text)
        timing.update({"parser": processed["timing_ms"]["parser"],
                       "qwen": processed["timing_ms"]["qwen"],
                       "server_total": int((time.time() - t_start) * 1000)})
        out = dict(base, ok=processed["status"] == "ok", processed=processed,
                   timing_ms=timing, error=None, vision_lease=list(self._lease_states))
        self._remember(out)
        return out

    def handle_text_command(self, text):
        t_start = time.time()
        self._lease_states = []
        self.counters["text_commands"] += 1
        processed = self.process_text(text)
        out = {"ok": processed["status"] == "ok", "text": text, "processed": processed,
               "qwen_device": "Modalix MLA", "vision_lease": list(self._lease_states),
               "timing_ms": {"parser": processed["timing_ms"]["parser"],
                             "qwen": processed["timing_ms"]["qwen"],
                             "server_total": int((time.time() - t_start) * 1000)}}
        self._remember(out)
        return out

    def debug_infer(self, which, wav_path=None):
        out = {"model": which, "timing_ms": {}}
        if which in ("whisper", "both"):
            wav = wav_path or write_silence_wav(os.path.join(TMP_DIR, "board_voice_debug.wav"), 1.0)
            t = time.time()
            with self.gate.hold("debug_whisper") as lease:
                r = self._run_whisper(wav, "ko")
            out["whisper_text"], out["vision_lease_whisper"] = (r.text or "")[:120], lease.state
            out["timing_ms"]["whisper"] = int((time.time() - t) * 1000)
        if which in ("qwen", "both"):
            t = time.time()
            with self.gate.hold("debug_qwen") as lease:
                raw, metrics = self._run_qwen(SELFTEST_TEXT)
            out["qwen_text"], out["qwen_metrics"], out["vision_lease_qwen"] = raw[:120], metrics, lease.state
            out["timing_ms"]["qwen"] = int((time.time() - t) * 1000)
        return out

    def _remember(self, out):
        processed = out.get("processed") or {}
        self.last_commands.append({"at": time.time(),
                                   "text": out.get("stt_text") or out.get("text"),
                                   "display": processed.get("display"),
                                   "source": processed.get("source"),
                                   "qwen_invoked": processed.get("qwen_invoked"),
                                   "reason": processed.get("reason") or out.get("refused"),
                                   "timing_ms": out.get("timing_ms")})
        del self.last_commands[:-50]


class Handler(BaseHTTPRequestHandler):
    server_version = "PaletteNeatDemoVoice/1.0"
    protocol_version = "HTTP/1.1"
    pipeline = None

    def log_message(self, fmt, *args):
        if self.path.startswith("/health"):
            return
        log("%s %s" % (self.address_string(), fmt % args))

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _locked(self, fn):
        if not self.pipeline.snapshot()["ready"]:
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "not ready"})
            return
        if not self.pipeline.inference_lock.acquire(blocking=False):
            self._send(HTTPStatus.CONFLICT, {"error": "busy"})
            return
        try:
            self._send(HTTPStatus.OK, fn())
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        finally:
            self.pipeline.inference_lock.release()

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in ("/health", "/status", "/"):
            self._send(HTTPStatus.OK, self.pipeline.snapshot())
        elif parsed.path == "/debug/infer":
            which = (query.get("model", ["both"])[0] or "both").lower()
            if which not in ("whisper", "qwen", "both"):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "model=%s" % which})
                return
            self._locked(lambda: self.pipeline.debug_infer(which))
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown path %s" % parsed.path})

    def _body(self, limit):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > limit:
            return None
        return self.rfile.read(length)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/text_command":
            body = self._body(64 * 1024)
            try:
                text = json.loads(body.decode("utf-8")).get("text") if body else None
            except (ValueError, AttributeError):
                text = None
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
                self._send(HTTPStatus.BAD_REQUEST,
                           {"ok": False, "error": "send {\"text\": \"...\"} (1-%d characters)"
                                                  % MAX_TEXT_CHARS})
                return
            self._locked(lambda: self.pipeline.handle_text_command(text.strip()))
            return
        if parsed.path != "/voice_command":
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown path %s" % parsed.path})
            return
        language = (parse_qs(parsed.query).get("language", ["ko"])[0] or "ko").lower()
        body = self._body(MAX_AUDIO_BYTES)
        if body is None or language not in LANGUAGE_MODES or not looks_like_wav(body):
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "need a WAV body and ko|en|auto"})
            return
        self._locked(lambda: self.pipeline.handle_voice_command(body, language))


def main():
    ap = argparse.ArgumentParser(description="Palette NEAT SDK Demo on-device voice runtime")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--config", default=os.path.join(BACKEND_DIR, "command_config.json"))
    ap.add_argument("--prompt", default=os.path.join(VOICE_DIR, "command_prompt.txt"))
    ap.add_argument("--arbiter", dest="arbiter", action="store_true", default=True,
                    help="take the vision MLA lease around every inference (default)")
    ap.add_argument("--no-arbiter", dest="arbiter", action="store_false")
    ap.add_argument("--arbiter-port", type=int, default=DEFAULT_ARBITER_PORT)
    args = ap.parse_args()

    if os.uname().machine != "aarch64":
        log("REFUSING to run on %s: this runtime executes Whisper-medium and Qwen3-0.6B on the "
            "Modalix MLA and must run on the DevKit." % os.uname().machine)
        return 2

    manifest = json.load(open(os.path.join(APP_DIR, "config", "model_manifest.json"), encoding="utf-8"))
    root = model_root()
    whisper_dir = os.path.join(root, manifest["models"]["whisper_medium"]["path"])
    qwen_dir = os.path.join(root, manifest["models"]["qwen3_0_6b"]["path"])
    for name, path in (("Whisper-medium", whisper_dir), ("Qwen3-0.6B", qwen_dir)):
        if not os.path.isdir(path):
            log("ERROR: required %s artifact missing: %s (config/model_manifest.json)" % (name, path))
            return 2
    os.makedirs(TMP_DIR, exist_ok=True)
    spec = CommandSpec(args.config, args.prompt)
    gate = VisionGate(args.arbiter_port, args.arbiter)
    log("Whisper-medium : %s  (pyneat genai.ASRModel, Modalix MLA)" % whisper_dir)
    log("Qwen3-0.6B     : %s  (pyneat genai.GenAIModel, Modalix MLA)" % qwen_dir)
    log("command config : %s (sha256 %s)" % (spec.config_path, spec.config_sha256[:16]))
    log("MLA arbitration: %s (arbiter 127.0.0.1:%d)" % ("ON" if gate.enabled else "OFF", gate.port))

    pipeline = BoardVoicePipeline(spec, whisper_dir, qwen_dir, gate)
    Handler.pipeline = pipeline
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            log("ERROR: port %d is already in use" % args.port)
            return 1
        raise
    httpd.daemon_threads = True
    log("listening on http://%s:%d" % (args.host, args.port))
    threading.Thread(target=pipeline.load, name="model-loader", daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
