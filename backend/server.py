#!/usr/bin/env python3
"""
server.py - the Palette NEAT SDK Demo UI web server.

    python3 backend/server.py                     # reads config/ui_config.json
    python3 backend/server.py --port 9910
    python3 backend/server.py --no-tls            # plain HTTP (no browser mic)

WHAT THIS PROCESS DOES

    serves    the demo page and its scripts
    serves    the INSTALLED Insight's overlay renderer, proxied not copied
    proxies   one WebRTC SDP round trip per panel to vf   (signalling only)
    owns      the panel state: mapping, detection ON/OFF, class list, box colour
    relays    a browser-recorded WAV, or a typed command, to the on-device voice
              runtime, which runs the ONE command path (backend/command_processor.py:
              parser first, Qwen3-0.6B fallback) and returns a validated command
    applies   that command after checking it against the command schema again

WHAT THIS PROCESS DOES NOT DO, BY CONSTRUCTION

    It never binds a video port, never binds a metadata port, never opens a UDP
    socket, never sees an H.264 byte and never sees a metadata datagram.  The
    DevKit sends video and metadata straight into Insight's own ingest sockets
    (udp 9000/9001 and 9100/9101) and the browser reads both out of Insight over
    WebRTC.  So there is no encoded-video proxy, no forwarding, no relay and no
    keyframe gate anywhere in this app - which is what makes "a normal command
    cannot interrupt the picture" a property of the architecture instead of a
    promise about the code (spec sections 7, 18, 25).

    grep -nE 'socket\\.socket|SOCK_DGRAM|7000|7100' backend/*.py

REQUEST MAP

    GET  /                       the demo page
    GET  /app.js /overlay.js /style.css
    GET  /insight/drawing.js     proxied from vf, no-store
    POST /offer?channel=N        proxied to vf; returns vf's status verbatim
    GET  /api/config             bootstrap: sources, panels, render settings, capture mode
    GET  /api/state              one panel-state snapshot
    GET  /api/events             server-sent events: state + command + voice status
    GET  /api/workloads          the four MLA workloads: YOLO fps, GenAI model state
    POST /api/command            one canonical command from the manual controls
    POST /api/command/clear      Clear button: empties the displayed Recognized / Command
                                 record only (no panel state, camera or model is touched)
    POST /api/text/command       {"text": ...}  -> the command path (same as voice)
    POST /api/transcript         alias of /api/text/command
    POST /api/voice/command?language=auto|ko|en
                                 body: 16 kHz mono WAV -> Whisper -> the command path
    GET  /api/voice/health       the voice runtime's state as we last saw it
    GET  /api/sources/live       is each source actually sending, per Insight
    GET  /api/vision/log         the CAMERA's own messages, classified,
                                 read-only from the vision log + dmesg
    GET  /api/diagnostics        Insight per-channel ingest counters
    GET  /api/health             this app
"""

import argparse
import json
import mimetypes
import os
import signal
import ssl
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
WEB_DIR = os.path.join(APP_DIR, "web")
CONFIG_PATH = os.path.join(APP_DIR, "config", "ui_config.json")

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import command_config                                            # noqa: E402
import command_normalizer                                        # noqa: E402
import command_processor                                         # noqa: E402
import insight_client                                            # noqa: E402
import panel_state                                               # noqa: E402
import vision_log                                                # noqa: E402
import voice_bridge                                              # noqa: E402

MAX_BODY_BYTES = 16 * 1024 * 1024        # a long WAV, bounded
# How long to watch the RTP packet counter before deciding a source is silent.
# At 30 fps this is hundreds of packets when the camera is running and exactly
# zero when it is not, which is the only unambiguous signal available.
SOURCE_LIVENESS_WINDOW_S = 0.35
# A handful of packets is not "sending": it is the tail of a stream stopping or
# the first moment of one resuming.  Measured over this window a running camera
# delivers 120-820 packets, and the tail of a stopping one delivered 9, so the
# floor sits far below normal and far above noise.  It matters because `live`
# is what tells a silent browser to re-offer, and re-offering during a pause is
# the one thing that must not happen.
SOURCE_LIVENESS_MIN_PACKETS = 20
SSE_KEEPALIVE_S = 15.0
STATIC_FILES = {
    "/app.js": "app.js",
    "/overlay.js": "overlay.js",
    "/webrtc.js": "webrtc.js",
    "/style.css": "style.css",
}


def log(message):
    print("[ui] %s" % message, flush=True)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def load_ui_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("web", "insight", "sources", "panels", "voice", "render"):
        if key not in data:
            raise ValueError("%s is missing %r" % (path, key))
    if set(data["panels"]) != {"left", "right"}:
        raise ValueError("%s: panels must define exactly left and right" % path)
    for position, source in data["panels"].items():
        if str(source) not in data["sources"]:
            raise ValueError("%s: panels.%s points at unknown source %r"
                             % (path, position, source))
    if len(set(str(v) for v in data["panels"].values())) != 2:
        raise ValueError("%s: both panels map to the same source" % path)
    resolve_insight_bases(data)
    return data


def resolve_insight_bases(config, rebuild=False):
    """Turn insight.host / api_port / vf_port into the two base URLs.

    Insight is not necessarily on the machine running this UI.  It lives in the
    Neat SDK container on the SDK host, so running this UI on the DevKit - which
    is a good idea, because a DevKit port really is on the LAN where a container
    port is not - means Insight is one hop away and 127.0.0.1 is the wrong
    answer.  Hence one `host` here rather than two hardcoded URLs.

    An api_base / vf_base written in the configuration file still wins: that is
    someone naming an exact URL rather than a host, so --insight-host does not
    overwrite it (`rebuild`).
    """
    insight = config["insight"]
    host = str(insight.get("host") or "127.0.0.1").strip()
    if host in ("local", "localhost", ""):
        host = "127.0.0.1"
    insight["host"] = host
    bracketed = "[%s]" % host if ":" in host else host
    insight.setdefault("api_port", 9900)
    insight.setdefault("vf_port", 8081)

    if "from_file" not in insight.setdefault("bases", {}):
        insight["bases"]["from_file"] = sorted(
            name for name in ("api_base", "vf_base") if insight.get(name))

    for name, port in (("api_base", insight["api_port"]),
                       ("vf_base", insight["vf_port"])):
        pinned = name in insight["bases"]["from_file"]
        if insight.get(name) and not (rebuild and not pinned):
            continue
        insight[name] = "https://%s:%d" % (bracketed, int(port))
    return insight


# --------------------------------------------------------------------------
# Finding Insight when it is not on this machine
# --------------------------------------------------------------------------
#
# Running this UI on the DevKit is the right thing to do - a DevKit port is a
# real port on the LAN where an SDK-container port is not - and then Insight is
# one hop away on the SDK host.  Getting `--insight-host` wrong produces a page
# with two empty panels, so rather than only documenting the flag, the machine
# is asked where the SDK host is and the answer is VERIFIED before it is used.
#
# Nothing is guessed.  Each candidate comes from something this machine already
# knows, and a candidate is only accepted if GET /api/health on it answers
# {"service": "neat-insight"}.  An address that merely has something listening
# on 9900 is rejected.

def insight_host_candidates():
    """[(address, why)] - places the SDK host could be, most authoritative first."""
    out = []

    def add(address, why):
        address = (address or "").strip()
        if not address or address.startswith("127.") or address in ("::1",):
            return
        if any(address == existing for existing, _ in out):
            return
        out.append((address, why))

    for name in ("NEAT_INSIGHT_HOST", "CONTAINER_HOST_IP", "NFS_SERVER_HOST_IP"):
        add(os.environ.get(name, ""), "$%s" % name)

    # /workspace is NFS-mounted from the SDK host, so the mount table names it.
    # This is the strongest signal available on a DevKit, and the mount that
    # actually contains this project is the strongest of those.
    ours, others = [], []
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 4 or fields[2] not in ("nfs", "nfs4"):
                    continue
                source, target, options = fields[0], fields[1], fields[3]
                server = source.split(":")[0] if ":" in source else ""
                for option in options.split(","):
                    if option.startswith("addr="):
                        server = option[len("addr="):] or server
                if not server:
                    continue
                entry = (server, "NFS server for %s" % target)
                holds_project = (target == "/"
                                 or APP_DIR.startswith(target.rstrip("/") + "/"))
                (ours if holds_project else others).append(entry)
    except OSError:
        pass
    for server, why in ours + others:
        add(server, why)

    # The default gateway: on this bench the SDK host is the DevKit's gateway.
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as handle:
            for line in handle.read().splitlines()[1:]:
                fields = line.split()
                if len(fields) < 3 or fields[1] != "00000000":
                    continue
                packed = int(fields[2], 16)
                add(".".join(str((packed >> shift) & 0xFF)
                             for shift in (0, 8, 16, 24)), "default gateway")
    except (OSError, ValueError):
        pass

    return [(address, why) for address, why in out if address]


def discover_insight_host(config, log_line=log):
    """Find a real Insight on the network, or return None.

    Called only when Insight was expected on this machine and is not there.
    Returns the address of an endpoint that identified itself as neat-insight.
    """
    candidates = insight_host_candidates()
    if not candidates:
        return None
    log_line("looking for Insight on this network (it is not on this machine)")
    api_port = int(config["insight"]["api_port"])
    for address, why in candidates:
        bracketed = "[%s]" % address if ":" in address else address
        base = "https://%s:%d" % (bracketed, api_port)
        probe = insight_client.InsightClient(base, base, timeout=2.5)
        if probe.identifies_as_insight():
            log_line("found neat-insight at %s  (%s)" % (base, why))
            return address
        log_line("  not Insight: %s  (%s)" % (base, why))
    return None


# --------------------------------------------------------------------------
# TLS
# --------------------------------------------------------------------------

def _looks_like_ip(text):
    import ipaddress
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def local_addresses():
    """This machine's non-loopback IPv4 addresses, for the certificate.

    Deliberately no socket: see host_ip() for why this process opens none.
    """
    import socket
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127.") and address not in out:
                out.append(address)
    except Exception:                                           # noqa: BLE001
        pass
    for env in ("CONTAINER_HOST_IP", "NFS_SERVER_HOST_IP"):
        value = os.environ.get(env, "").strip()
        if value and _looks_like_ip(value) and value not in out:
            out.append(value)
    return out


def resolve_tls(config):
    """(cert, key) or None.

    The browser microphone needs a secure context, so HTTPS is the default.
    First choice is the SDK's own certificate - the one Insight itself is served
    with, per ~/.insight-config/neat-port-map.json - so a browser that already
    trusts Insight trusts this too and there is no second warning to click
    through.  Failing that, one self-signed certificate is generated INSIDE this
    project and reused.  Nothing outside this project is written or changed.
    """
    tls = config["web"].get("tls") or {}
    if not tls.get("enabled", True):
        return None

    cert, key = tls.get("cert"), tls.get("key")
    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        if os.access(cert, os.R_OK) and os.access(key, os.R_OK):
            log("TLS: using the SDK certificate %s" % cert)
            return cert, key
        log("TLS: %s exists but is not readable; generating our own" % cert)

    if not tls.get("generate_if_missing", True):
        raise SystemExit("TLS is enabled but %r / %r are unusable and "
                         "generate_if_missing is false" % (cert, key))

    out_dir = os.path.join(APP_DIR, tls.get("generated_dir", "runtime/tls"))
    os.makedirs(out_dir, exist_ok=True)
    own_cert = os.path.join(out_dir, "cert.pem")
    own_key = os.path.join(out_dir, "key.pem")
    if os.path.isfile(own_cert) and os.path.isfile(own_key):
        log("TLS: using this project's certificate %s" % own_cert)
        return own_cert, own_key

    # Every address a browser might use, so the only thing left to click through
    # is the unknown issuer - not that plus a name mismatch. getUserMedia needs
    # a secure context, and an overridden certificate error still gives one, but
    # two warnings during a demo is one too many.
    names = ["DNS:localhost", "IP:127.0.0.1"]
    for address in local_addresses():
        entry = "IP:%s" % address
        if entry not in names:
            names.append(entry)
    for extra in tls.get("extra_names") or []:
        entry = ("IP:%s" % extra) if _looks_like_ip(extra) else ("DNS:%s" % extra)
        if entry not in names:
            names.append(entry)

    log("TLS: generating a self-signed certificate in %s" % out_dir)
    log("TLS: valid for %s" % ", ".join(names))
    result = subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", own_key, "-out", own_cert, "-days", "825",
         "-subj", "/CN=whisper-yolo-ui",
         "-addext", "subjectAltName=" + ",".join(names)],
        capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(own_cert):
        raise SystemExit("could not generate a certificate:\n%s\n"
                         "Run with --no-tls to serve plain HTTP (the browser "
                         "microphone will not work)." % result.stderr.strip())
    os.chmod(own_key, 0o600)
    return own_cert, own_key


# --------------------------------------------------------------------------
# Application state
# --------------------------------------------------------------------------

def read_capture_selection(path):
    """The capture mode run.sh's preflight selected, from its selection file.

    The file is written by the vision binary (--selection-out) and read by
    scripts/start_pipeline.sh, so this is the mode the running pipeline uses -
    never a hard-coded value. None when there is no such file (--no-vision).
    """
    if not path or not os.path.isfile(path):
        return None
    import re
    import shlex
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                try:
                    parts = shlex.split(raw)
                except ValueError:
                    parts = [raw]
                values[key.strip()] = parts[0] if parts else ""
    except OSError:
        return None
    mode = values.get("USB_SELECTED_MODE", "")
    match = re.match(r"^(?:(\w+):)?(\d+)x(\d+)@(\d+(?:\.\d+)?)$", mode)
    if not match:
        return None
    fmt, width, height, fps = match.group(1), int(match.group(2)), int(match.group(3)), float(match.group(4))
    cameras = {}
    for index in ("0", "1"):
        prefix = "USB_SELECTED_CAMERA%s_" % index
        cameras[index] = {key[len(prefix):].lower(): value
                          for key, value in values.items() if key.startswith(prefix)}
    return {"mode": mode, "format": fmt, "width": width, "height": height, "fps": fps,
            "label": "%dp%g" % (height, fps), "cameras": cameras,
            "preferred_mode": values.get("USB_SELECTED_PREFERRED_MODE")}


class LastCommand:
    """The one command row the UI shows: what was said or typed and what it became.

    Held separately from PanelState on purpose: a command that is not understood
    updates this and nothing else, so the display record and the render state
    cannot share a version number.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.version = 1
        self._value = {
            "input": None, "origin": None, "command": None, "display": None,
            "status": None, "reason": None, "source": None, "qwen_used": False,
            "qwen_invoked": False, "result": None, "language": None,
            "detected_language": None, "timing_ms": None, "at": None,
        }

    def snapshot(self):
        with self._lock:
            out = dict(self._value)
            out["version"] = self.version
            return out

    def set(self, **fields):
        with self._lock:
            value = {key: None for key in self._value}
            value.update({k: v for k, v in fields.items() if k in value})
            value["at"] = time.time()
            self._value = value
            self.version += 1
            return self.version


class Broadcaster:
    """Fan-out to every open /api/events stream.

    Each subscriber gets its own bounded list and its own Event.  A slow or dead
    browser therefore cannot block a command: the write happens on the
    subscriber's own request thread, and a full buffer drops the oldest frame
    rather than waiting.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []

    def subscribe(self):
        entry = {"pending": [], "event": threading.Event(), "alive": True}
        with self._lock:
            self._subscribers.append(entry)
        return entry

    def unsubscribe(self, entry):
        entry["alive"] = False
        with self._lock:
            if entry in self._subscribers:
                self._subscribers.remove(entry)
        entry["event"].set()

    def publish(self, payload):
        blob = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            targets = list(self._subscribers)
        for entry in targets:
            pending = entry["pending"]
            if len(pending) > 32:
                del pending[:-8]
            pending.append(blob)
            entry["event"].set()

    @property
    def count(self):
        with self._lock:
            return len(self._subscribers)


class App:
    def __init__(self, config, capture_selection=None):
        self.config = config
        self.capture_selection = capture_selection
        self.command_config = command_config.load()
        self.state = panel_state.PanelState(
            self.command_config,
            panels={k: str(v) for k, v in config["panels"].items()})
        self.normalizer = command_normalizer.CommandNormalizer(self.command_config)
        self.insight = insight_client.InsightClient(
            config["insight"]["api_base"], config["insight"]["vf_base"])
        voice = config["voice"]
        self.voice = voice_bridge.VoiceServerClient(
            voice["host"], voice["port"],
            request_timeout=voice.get("request_timeout_s", 120),
            health_timeout=voice.get("health_timeout_s", 4))
        self.voice_health = voice_bridge.HealthMonitor(self.voice)
        self.last_command = LastCommand()
        self.events = Broadcaster()
        self.started_at = time.time()
        self._voice_watch = None
        self._liveness = None
        self._liveness_at = 0.0
        self._liveness_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start_background(self):
        self.voice_health.start()
        self._voice_watch = threading.Thread(target=self._voice_watch_loop,
                                             name="voice-watch", daemon=True)
        self._voice_watch.start()

    def stop_background(self):
        self.voice_health.stop()

    def _voice_watch_loop(self):
        """Push a voice-status frame whenever the DevKit's state changes."""
        last = None
        while True:
            snapshot = self.voice_health.snapshot()
            key = (snapshot.get("connection"), snapshot.get("ready"),
                   snapshot.get("state"))
            if key != last:
                last = key
                self.events.publish({"kind": "voice", "voice": snapshot})
            time.sleep(1.0)

    # -- the one place a command is applied --------------------------------

    def apply_processed(self, processed, origin, extra=None):
        """Apply one command_processor result and publish exactly what changed.

        Voice, typed text and the manual controls all end here. The command is
        checked against the schema once more before PanelState sees it; anything
        that is not a valid command becomes a no-op and "Command not understood".
        """
        processed = processed or {}
        command = processed.get("command")
        problem = None
        if processed.get("status") == "ok":
            manual_auto = (isinstance(command, dict) and origin == "manual"
                           and command.get("action") == "set_box_color"
                           and command.get("color") == panel_state.AUTO_COLOR)
            if not manual_auto:
                problem = command_normalizer.check_command(command, self.command_config)
        understood = processed.get("status") == "ok" and problem is None
        if not understood:
            command = {"action": "unknown", "original_text": processed.get("text") or ""}
        result = self.state.apply_command(command)
        applied = understood and result.ok
        if applied:
            display = processed.get("display") or command_processor.describe(command)
        elif understood:
            display = result.message
        else:
            display = command_processor.NOT_UNDERSTOOD
        record = {
            "input": processed.get("text"),
            "origin": origin,
            "command": command,
            "display": display,
            "status": "ok" if applied else ("rejected" if understood else "not_understood"),
            "reason": problem or processed.get("reason") or (None if applied else result.message),
            "source": processed.get("source") if applied else None,
            "qwen_used": bool(applied and processed.get("source") == "qwen3"),
            "qwen_invoked": bool(processed.get("qwen_invoked")),
            "result": result.as_dict(),
            "timing_ms": processed.get("timing_ms"),
        }
        record.update(extra or {})
        self.last_command.set(**record)
        log("%s command %r -> %s (%s)" % (origin, processed.get("text"), display,
                                          record["reason"] or processed.get("source")))
        payload = {"kind": "command", "command": self.last_command.snapshot(),
                   "parser": processed.get("parser")}
        # A command that changed nothing carries no state frame at all.
        if result.changed:
            payload["state"] = self.state.snapshot()
        self.events.publish(payload)
        return record

    def publish_error(self, origin, text, display, reason, extra=None):
        """A request that produced no command (busy, no speech, unreachable)."""
        record = {"input": text, "origin": origin, "command": None, "display": display,
                  "status": "error", "reason": reason}
        record.update(extra or {})
        self.last_command.set(**record)
        log("%s request: %s (%s)" % (origin, display, reason))
        self.events.publish({"kind": "command", "command": self.last_command.snapshot()})
        return record

    def clear_display(self):
        """The Clear button: the Recognized / Command readout returns to its placeholder.

        Display only. It empties LastCommand, which is the display record and nothing
        else; the panel state (cameras, detection, classes, colours, LEFT/RIGHT mapping)
        is PanelState, a different object, and nothing is sent to the voice runtime or
        the vision process.
        """
        self.last_command.set()
        snapshot = self.last_command.snapshot()
        log("display cleared (Recognized / Command)")
        self.events.publish({"kind": "command", "command": snapshot, "cleared": True})
        return snapshot

    def apply_canonical(self, command, origin="manual"):
        """A command that is already canonical (the manual controls)."""
        processed = {"text": None, "status": "ok", "command": command, "source": "manual",
                     "display": command_processor.describe(command)}
        return self.apply_processed(processed, origin)

    def run_text_command(self, text):
        """A typed command: the same command path a recognised utterance takes.

        The on-device voice runtime owns Qwen3, so it runs command_processor.process()
        there. If it cannot be reached, the same function runs here with no Qwen3
        (parser only) and says so.
        """
        text = (text or "").strip()
        if not text:
            return self.publish_error("text", text, "Type a command first", "empty text")
        extra = {}
        try:
            reply, status = self.voice.text_command(text)
        except voice_bridge.VoiceError as exc:
            reply, status = None, None
            extra["reason_note"] = str(exc)
        if status == 409:
            return self.publish_error("text", text, "Busy – try again in a moment",
                                      "the voice runtime is processing another command")
        if status == 200 and isinstance(reply, dict) and reply.get("processed"):
            return self.apply_processed(reply["processed"], "text",
                                        {"timing_ms": reply.get("timing_ms")})
        processed = command_processor.process(text, self.normalizer, None)
        if processed["status"] != "ok":
            processed["reason"] = "%s (voice runtime unavailable: HTTP %s)" % (
                processed["reason"], status)
        return self.apply_processed(processed, "text")

    # -- snapshots ---------------------------------------------------------

    def client_config(self):
        sources = {}
        for key, entry in self.config["sources"].items():
            sources[key] = dict(entry)
            sources[key]["classes"] = self.command_config.stream_classes(key)
        return {
            "sources": sources,
            "panels": {k: str(v) for k, v in self.config["panels"].items()},
            "render": self.config["render"],
            "insight": {
                "video_udp_base": self.config["insight"]["video_udp_base"],
                "metadata_udp_base": self.config["insight"]["metadata_udp_base"],
                "api_base": self.config["insight"]["api_base"],
                "vf_base": self.config["insight"]["vf_base"],
            },
            "voice": {"endpoint": self.voice.base,
                      "languages": list(voice_bridge.LANGUAGES),
                      "default_language": self.default_language()},
            "capture": read_capture_selection(self.capture_selection),
            "workloads": self.workload_catalog(),
            "canonical_objects": self.command_config.canonical_objects,
            "colors": self.command_config.colors,
            "actions": sorted(command_config.SUPPORTED_ACTIONS),
        }

    def default_language(self):
        language = self.config["voice"].get("language", "ko")
        return language if language in voice_bridge.LANGUAGES else "ko"

    def workload_catalog(self):
        """The four AI workloads, from config/model_manifest.json (all Modalix MLA)."""
        with open(os.path.join(APP_DIR, "config", "model_manifest.json"),
                  encoding="utf-8") as handle:
            models = json.load(handle)["models"]
        names = (("yolo26m", "YOLO26m Detection"), ("yolo26m_seg", "YOLO26m Segmentation"),
                 ("whisper_medium", "Whisper-medium"), ("qwen3_0_6b", "Qwen3-0.6B"))
        return [{"key": key, "name": name, "runtime": models[key]["runtime"],
                 "device": models[key]["device"]} for key, name in names]

    def workloads(self):
        """Live state of the four MLA workloads for the header chips."""
        stats = vision_log.latest_stats()
        voice = self.voice_health.snapshot()
        models = voice.get("models") or {}
        return {
            "vision": {channel: {"task": entry.get("task"),
                                 "yolo_fps": entry.get("yolo_fps")}
                       for channel, entry in stats.items()},
            "voice": {"ready": voice.get("ready"), "state": voice.get("state"),
                      "busy": voice.get("busy"), "counters": voice.get("counters"),
                      "whisper_loaded": bool((models.get("whisper") or {}).get("loaded")),
                      "qwen_loaded": bool((models.get("qwen") or {}).get("loaded"))},
        }

    def source_liveness(self, max_age_s=1.0):
        """Is each source actually SENDING right now, per Insight's own counters.

        This exists to keep the browser from "fixing" something that is not
        broken.  A panel can go black for two completely different reasons:

          the source stopped sending     nothing to do but wait.  The pipeline
                                         stops the cameras deliberately for the
                                         duration of a Whisper/Qwen inference
                                         (`pause_behaviour: stop-camera`), and
                                         that is measured at 2.2-3.7 s per
                                         spoken command.
          the source is sending but we   our WebRTC peer was displaced - vf
          receive nothing                gives a channel's media to ONE peer -
                                         and re-offering is the correct repair.

        A browser cannot tell those apart: both look like silence.  This
        endpoint can, because it reads the UDP arriving at vf, so the page asks
        before it touches a connection.  Cached briefly: several panels ask at
        once and Insight should not be polled once per panel per second.
        """
        now = time.monotonic()
        with self._liveness_lock:
            if self._liveness and now - self._liveness_at < max_age_s:
                return self._liveness

        # Two readings, a moment apart, and the answer is whether the packet
        # COUNTER moved.  Measured the hard way: neither of the obvious fields
        # can see a short pause.
        #
        #   channel `active`  vf holds it true for a 3 s TTL, and the pause is
        #                     2.2-3.7 s, so it usually never flips.
        #   `bitrate_bps`     a smoothed average.  Through a real 2 s camera
        #                     stop it read 0.51 Mbit/s, not 0.
        #
        # A counter that has not advanced is unambiguous. The cost is one short
        # sleep, paid only when a panel is already silent and asking why.
        first = self.insight.channel_summary()
        time.sleep(SOURCE_LIVENESS_WINDOW_S)
        second = self.insight.channel_summary()

        out = {"sources": {},
               "error": second.get("error") or first.get("error"),
               "window_s": SOURCE_LIVENESS_WINDOW_S}
        for key, entry in self.config["sources"].items():
            channel = str(entry["channel"])
            before = first.get(channel) or {}
            after = second.get(channel) or {}
            packets_before = before.get("packets_received")
            packets_after = after.get("packets_received")
            delta = 0
            if isinstance(packets_before, int) and isinstance(packets_after, int):
                delta = max(0, packets_after - packets_before)
            out["sources"][key] = {
                "channel": entry["channel"],
                "live": delta >= SOURCE_LIVENESS_MIN_PACKETS,
                "packets_delta": delta,
                "min_packets": SOURCE_LIVENESS_MIN_PACKETS,
                "bitrate_bps": after.get("bitrate_bps") or 0,
                "active": bool(after.get("active")),
                "track_attached": after.get("track_attached"),
                # The displacement evidence.  vf attaches a channel's video to
                # ONE peer, so `peers > 1` is what being displaced looks like -
                # and it is the only condition under which re-offering repairs
                # anything.  Silence with `peers <= 1` means we ARE the
                # channel's peer and the frames are simply not there yet, which
                # is what an upstream camera pause looks like; re-offering then
                # would blank a panel that was about to come back on its own.
                "peers": after.get("peers"),
            }
        with self._liveness_lock:
            self._liveness = out
            self._liveness_at = time.monotonic()
        return out

    def full_snapshot(self):
        return {"kind": "snapshot",
                "state": self.state.snapshot(),
                "command": self.last_command.snapshot(),
                "voice": self.voice_health.snapshot()}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "DroneDemoUI/1.0"
    protocol_version = "HTTP/1.1"

    app = None                       # set by serve()

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/events"):
            return
        log("%s %s" % (self.address_string(), fmt % args))

    def _send(self, status, body, content_type="application/json",
              extra_headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status, payload, extra_headers=None):
        self._send(status, json.dumps(payload, ensure_ascii=False),
                   "application/json; charset=utf-8", extra_headers)

    def _error(self, status, message):
        self._send_json(status, {"error": message})

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            length = int(length)
        except ValueError:
            return b""
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("body is %s bytes; the limit is %d"
                             % (length, MAX_BODY_BYTES))
        return self.rfile.read(length)

    def _read_json(self):
        body = self._read_body()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    # -- GET ---------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                return self._serve_web_file("index.html", "text/html; charset=utf-8")
            if path in STATIC_FILES:
                return self._serve_web_file(STATIC_FILES[path])
            if path == "/insight/drawing.js":
                return self._serve_drawing_js()
            if path == "/api/config":
                return self._send_json(HTTPStatus.OK, self.app.client_config())
            if path == "/api/state":
                return self._send_json(HTTPStatus.OK, self.app.full_snapshot())
            if path == "/api/events":
                return self._serve_events()
            if path == "/api/voice/health":
                health = dict(self.app.voice_health.snapshot())
                health["languages"] = list(voice_bridge.LANGUAGES)
                health["default_language"] = self.app.default_language()
                return self._send_json(HTTPStatus.OK, health)
            if path == "/api/workloads":
                return self._send_json(HTTPStatus.OK, self.app.workloads(),
                                       {"Cache-Control": "no-store"})
            if path == "/api/sources/live":
                return self._send_json(HTTPStatus.OK, self.app.source_liveness(),
                                       {"Cache-Control": "no-store"})
            if path == "/api/vision/log":
                # READ-ONLY view of the camera pipeline's own messages and the
                # kernel's CSI-2 / VDMA messages.  This process never sees a
                # frame, so its own log cannot show them; vision_log.py reads
                # this project's vision log and dmesg, and classifies
                # every line so the benign libcamera noise is not mistaken for
                # the fault.  Nothing is written anywhere.
                query = parse_qs(parsed.query)
                include_all = (query.get("all") or ["0"])[0] not in ("0", "false", "")
                try:
                    limit = min(400, max(1, int((query.get("limit") or ["60"])[0])))
                except ValueError:
                    limit = 60
                return self._send_json(
                    HTTPStatus.OK,
                    vision_log.report(limit=limit, include_benign=include_all),
                    {"Cache-Control": "no-store"})
            if path == "/api/diagnostics":
                return self._send_json(HTTPStatus.OK, {
                    "insight_channels": self.app.insight.channel_summary(),
                    "subscribers": self.app.events.count,
                    "uptime_s": round(time.time() - self.app.started_at, 1)})
            if path == "/api/health":
                return self._send_json(HTTPStatus.OK, {
                    "service": "palette_neat_sdk_demo", "status": "ok",
                    "uptime_s": round(time.time() - self.app.started_at, 1)})
            if path == "/favicon.ico":
                return self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return self._error(HTTPStatus.NOT_FOUND, "no such path %r" % path)
        except Exception as exc:                                # noqa: BLE001
            log("GET %s failed: %s" % (path, exc))
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_web_file(self, name, content_type=None):
        full = os.path.join(WEB_DIR, name)
        if not os.path.isfile(full):
            return self._error(HTTPStatus.NOT_FOUND, "missing %s" % name)
        with open(full, "rb") as handle:
            body = handle.read()
        if content_type is None:
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type.endswith("javascript"):
                content_type += "; charset=utf-8"
        return self._send(HTTPStatus.OK, body, content_type,
                          {"Cache-Control": "no-store"})

    def _serve_drawing_js(self):
        """The installed Insight's renderer.

        `no-store` is deliberate: a browser holding an old drawing.js while the
        installed Insight has a new one is a documented Insight failure mode,
        and serving it through here with no caching removes it.
        """
        try:
            body = self.app.insight.drawing_js()
        except insight_client.InsightError as exc:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                               "Insight's overlay renderer is unavailable: %s. "
                               "No substitute renderer is used." % exc)
        return self._send(HTTPStatus.OK, body,
                          "application/javascript; charset=utf-8",
                          {"Cache-Control": "no-store"})

    def _serve_events(self):
        entry = self.app.events.subscribe()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self._write_event(self.app.full_snapshot())
            while entry["alive"]:
                if not entry["pending"]:
                    if not entry["event"].wait(SSE_KEEPALIVE_S):
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                entry["event"].clear()
                while entry["pending"]:
                    self._write_raw_event(entry["pending"].pop(0))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.events.unsubscribe(entry)
        self.close_connection = True

    def _write_event(self, payload):
        self._write_raw_event(json.dumps(payload, ensure_ascii=False))

    def _write_raw_event(self, blob):
        self.wfile.write(b"data: " + blob.encode("utf-8") + b"\n\n")
        self.wfile.flush()

    # -- POST --------------------------------------------------------------

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/offer":
                return self._offer(parse_qs(parsed.query))
            if path == "/api/command/clear":
                return self._send_json(HTTPStatus.OK, {"command": self.app.clear_display()})
            if path == "/api/command":
                return self._command()
            if path in ("/api/text/command", "/api/transcript"):
                return self._text_command()
            if path == "/api/voice/command":
                return self._voice_command(parse_qs(parsed.query))
            return self._error(HTTPStatus.NOT_FOUND, "no such path %r" % path)
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:                                # noqa: BLE001
            log("POST %s failed: %s" % (path, exc))
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _offer(self, query):
        """WebRTC signalling, passed through to vf unchanged.

        No media crosses this function.  vf's status codes carry meaning to the
        viewer - 503 "not identified yet, retry", 415 "this browser has no
        decoder, never retry" - so they are returned exactly as vf sent them.
        """
        try:
            channel = int((query.get("channel") or ["-1"])[0])
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "channel must be an integer")
        allowed = {int(entry["channel"])
                   for entry in self.app.config["sources"].values()}
        if channel not in allowed:
            return self._error(HTTPStatus.BAD_REQUEST,
                               "channel %d is not one of this demo's sources %s"
                               % (channel, sorted(allowed)))
        body = self._read_body()
        if not body:
            return self._error(HTTPStatus.BAD_REQUEST, "empty SDP offer")
        try:
            answer, status = self.app.insight.offer(channel, body)
        except insight_client.InsightError as exc:
            return self._error(HTTPStatus.BAD_GATEWAY,
                               "Insight vf is unreachable: %s" % exc)
        return self._send(status, answer, "application/json; charset=utf-8",
                          {"Cache-Control": "no-store"})

    def _command(self):
        payload = self._read_json()
        command = payload.get("command")
        if not isinstance(command, dict):
            return self._error(HTTPStatus.BAD_REQUEST,
                               "send {\"command\": {\"action\": ...}}")
        origin = payload.get("origin") or "manual"
        record = self.app.apply_canonical(command, origin)
        return self._send_json(HTTPStatus.OK,
                               {"command": record,
                                "state": self.app.state.snapshot()})

    def _text_command(self):
        """A typed command. Same command path as a recognised utterance."""
        payload = self._read_json()
        text = payload.get("text")
        if not isinstance(text, str) or len(text) > 300:
            return self._error(HTTPStatus.BAD_REQUEST,
                               "send {\"text\": \"...\"} (at most 300 characters)")
        record = self.app.run_text_command(text)
        return self._send_json(HTTPStatus.OK, {"command": record,
                                               "state": self.app.state.snapshot()})

    def _voice_command(self, query):
        """One spoken command: WAV in -> Whisper -> the command path -> record out."""
        language = (query.get("language") or [self.app.default_language()])[0]
        if language not in voice_bridge.LANGUAGES:
            return self._error(HTTPStatus.BAD_REQUEST,
                               "language must be one of %s" % list(voice_bridge.LANGUAGES))
        wav = self._read_body()
        if len(wav) < 64:
            return self._error(HTTPStatus.BAD_REQUEST, "no audio in the request body")
        self.app.events.publish({"kind": "activity", "activity": "voice-start"})
        try:
            reply, status = self.app.voice.voice_command(wav, language)
        except voice_bridge.VoiceError as exc:
            record = self.app.publish_error("voice", None, "Voice runtime not reachable",
                                            str(exc), {"language": language})
            return self._send_json(HTTPStatus.BAD_GATEWAY, {"command": record})
        finally:
            self.app.events.publish({"kind": "activity", "activity": "voice-end"})

        extra = {"language": language, "timing_ms": reply.get("timing_ms"),
                 "detected_language": reply.get("detected_language")}
        stt = (reply.get("stt_text") or "").strip()
        if status == 409:
            record = self.app.publish_error("voice", None, "Busy – try again in a moment",
                                            "the voice runtime is processing another command",
                                            extra)
        elif status != 200 or (not stt and not reply.get("processed")):
            record = self.app.publish_error(
                "voice", stt, "No speech recognised",
                reply.get("error") or "HTTP %s from the voice runtime" % status, extra)
        elif reply.get("refused"):
            record = self.app.publish_error("voice", stt, "Speech not clear – please repeat",
                                            reply["refused"], extra)
        else:
            record = self.app.apply_processed(reply.get("processed"), "voice", extra)
        return self._send_json(HTTPStatus.OK,
                               {"voice_status": status, "voice_reply": reply,
                                "command": record, "state": self.app.state.snapshot()})


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

# The ports the Neat SDK publishes from the container to the host and the LAN.
#
#   neat --json -> exposedPorts
#     tcp 9900 mainUI, 9999 codeUI, 10000 codeUIHttps, 8081 videoUI,
#     8554 rtsp.tcp, 8022 webSSH
#     udp 9000-9003 videoUDP, 9100-9103 metadataUDP, 40000-40007 webRTC
#
# A container port that is NOT on this list is reachable at 127.0.0.1 from
# INSIDE the container and from nowhere else - which is the whole of the
# "https://127.0.0.1:9910 cannot be opened" report.  Verified by measurement:
# a listener on 8022 in the container answered http://10.42.0.1:8022 from the
# DevKit; the same listener on 9910 was refused.
PUBLISHED_CONTAINER_TCP_PORTS = (8022, 8081, 8554, 9900, 9999, 10000)


def in_sdk_container():
    """Is this process inside the Neat SDK container rather than on the host?

    Two independent signs, both of which the container has and the SDK host
    does not: the installed Insight's venv path, and the container's own
    certificate directory.  No docker socket is consulted and nothing is
    executed - this only ever changes what the banner SAYS.
    """
    return (os.path.isdir("/opt/neat-insight/venv")
            and os.path.isdir("/sdk-cert")
            and os.environ.get("NFS_SERVER_HOST_IP", "") != "")


def reachability(port):
    """Where a browser can actually open this UI, as a list of lines."""
    lines = []
    if not in_sdk_container():
        lines.append("  This is not the SDK container, so any port is "
                     "reachable from the LAN.")
        return lines
    host = os.environ.get("NFS_SERVER_HOST_IP", "").strip() or "the SDK host"
    if port in PUBLISHED_CONTAINER_TCP_PORTS:
        lines.append("  Running INSIDE the SDK container, on published port "
                     "%d, so:" % port)
        lines.append("    from the SDK host      : https://127.0.0.1:%d  and "
                     "https://%s:%d" % (port, host, port))
        lines.append("    from another machine   : https://%s:%d" % (host, port))
        lines.append("    from in the container  : https://127.0.0.1:%d" % port)
    else:
        lines.append("  WARNING: port %d is NOT published by the SDK container."
                     % port)
        lines.append("    The SDK publishes only tcp %s."
                     % ", ".join(str(p) for p in
                                 PUBLISHED_CONTAINER_TCP_PORTS))
        lines.append("    So https://127.0.0.1:%d works ONLY from inside this"
                     % port)
        lines.append("    container. A browser on %s or on a laptop will get"
                     % host)
        lines.append("    'connection refused' - it is not a TLS or server "
                     "fault.")
        lines.append("    Use a published port:   ./run.sh --port 8022")
        lines.append("    or run this on the SDK host, where any port works.")
    return lines


def host_ip():
    """A browser-reachable address for this machine, for the banner only.

    Insight answers the same question from CONTAINER_HOST_IP (see
    neat_insight/app.py's /api/server-ip), so that is asked first and the
    hostname is only a fallback.  Deliberately no socket of any kind: the
    familiar "connect a UDP socket and read its local address" trick would put
    a datagram socket in this process, and this process having no datagram
    socket is a property the self-test checks.
    """
    import socket
    for env in ("CONTAINER_HOST_IP", "NFS_SERVER_HOST_IP"):
        value = os.environ.get(env, "").strip()
        if value:
            return value
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                return address
    except Exception:                                           # noqa: BLE001
        pass
    return "127.0.0.1"


def banner(scheme, port, app, tls_note):
    ip = host_ip()
    insight = app.config["insight"]
    # The two addresses the old banner ran together.  `ip` is where the BROWSER
    # opens this page; `insight_host` is where the video actually has to be sent
    # and where the renderer is fetched from.  They are the same machine only
    # when this UI runs on the Insight host.
    insight_host = insight["host"]
    if insight_host == "127.0.0.1":
        insight_host = ip
    same_machine = insight["host"] == "127.0.0.1"

    print("")
    print("=" * 66)
    print("  Palette NEAT SDK Demo UI:")
    print("    %s://%s:%d" % (scheme, ip, port))
    print("    %s://127.0.0.1:%d" % (scheme, port))
    print("=" * 66)
    print("  %s" % tls_note)
    print("")
    for line in reachability(port):
        print(line)
    print("")
    print("  This Demo UI            : %s, port %d" % (ip, port))
    print("  Insight Web UI          : %s   (unchanged, still running)"
          % insight["api_base"])
    print("  Insight vf viewer       : %s" % insight["vf_base"])
    if not same_machine:
        print("  Insight runs on         : %s   (not this machine%s)"
              % (insight["host"],
                 "; auto-detected" if insight.get("autodetected") else ""))
    print("")
    print("  Send video and metadata to the INSIGHT host, %s:" % insight_host)
    print("    video    udp %s:%d (ch0)   udp %s:%d (ch1)"
          % (insight_host, insight["video_udp_base"],
             insight_host, insight["video_udp_base"] + 1))
    print("    metadata udp %s:%d (ch0)   udp %s:%d (ch1)"
          % (insight_host, insight["metadata_udp_base"],
             insight_host, insight["metadata_udp_base"] + 1))
    # Where voice is looked for, and whether it is actually there.  A silent
    # "Voice server disconnected" in the browser was mistaken for a broken
    # voice server when the real answer was "this UI is on a host that cannot
    # reach the one configured", so the banner answers that question up front.
    print("  Voice    : %s" % app.voice.base)
    try:
        health = app.voice.health()
        if health.get("declares_camera_pause") is False:
            where = ("ON the vision board (MLA) - voice commands do NOT stop "
                     "the cameras")
        elif health.get("runs_on_vision_board") is False:
            where = "OFF the vision board - the cameras do not pause"
        else:
            where = "ON the vision board - each command may pause the cameras"
        print("             reachable, state=%s   (%s)"
              % (health.get("state"), where))
    except voice_bridge.VoiceError as exc:
        print("             NOT REACHABLE from this host: %s" % exc)
        print("")
        print("  The page will load and the pictures will work; only the")
        print("  microphone will be unavailable (\"Voice server disconnected\").")
        if app.voice.host in ("127.0.0.1", "localhost", "::1"):
            print("  It is configured on THIS host's loopback, which is right on the")
            print("  DevKit: ./run.sh starts the voice runtime there before this UI.")
            print("  Typed commands still work with the deterministic parser alone.")
        else:
            print("  Check that something answers there:")
            print("      curl -s %s/health" % app.voice.base)
        print("  Or point this UI at another one:  ./run.sh --voice-host <ip>")
    print("")
    print("  This process binds ONE TCP port and no UDP port at all.")
    print("  Video and metadata never pass through it.")
    print("")
    ok, lines = app.insight.preflight()
    print("  Insight preflight:")
    for line in lines:
        print("    " + line)
    if not ok:
        print("")
        if same_machine:
            print("  Insight is not on THIS machine (127.0.0.1 refused the")
            print("  connection). Neat Insight runs in the Neat SDK container on")
            print("  the SDK host, so if this UI is on the DevKit, say where:")
            print("")
            print("      ./run.sh --insight-host <sdk-host-ip>       # e.g. 10.42.0.1")
            print("")
            print("  or set insight.host in config/ui_config.json. Check it first")
            print("  with:  curl -k https://<sdk-host-ip>:%d/api/health"
                  % insight["api_port"])
        else:
            print("  Insight did not answer at %s. Check that it is running"
                  % insight["host"])
            print("  (insight-admin status on that machine) and that ports %d and"
                  % insight["api_port"])
            print("  %d are reachable from here." % insight["vf_port"])
        print("")
        print("  The page will load but the panels cannot connect until Insight")
        print("  answers. Nothing here substitutes a renderer or a transport.")
    print("")
    print("  Start the pipeline on the DevKit, aimed at the Insight host:")
    # This project's own launcher starts vision, the on-device voice runtime
    # and this UI together.
    print("    cd %s && ./run.sh   (INSIGHT_HOST=%s, ports %d/%d)"
          % (APP_DIR, insight_host, insight["video_udp_base"], insight["metadata_udp_base"]))
    print("")


def serve(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--port", type=int,
                        help="override web.port from the configuration")
    parser.add_argument("--bind", help="override web.bind")
    parser.add_argument("--insight-host",
                        help="the machine running Neat Insight (default: "
                             "insight.host in the configuration). Use this when "
                             "the UI runs somewhere Insight does not, e.g. the "
                             "UI on the DevKit and Insight on the SDK host")
    parser.add_argument("--no-insight-autodetect", action="store_true",
                        help="do not look for Insight elsewhere on the network "
                             "when it is not on this machine")
    parser.add_argument("--voice-host", help="override voice.host")
    parser.add_argument("--voice-port", type=int, help="override voice.port")
    parser.add_argument("--sync-buffer-ms", type=int,
                        help="override render.video_sync_buffer_ms - the "
                             "deliberate latency the browser adds so boxes "
                             "land on the frame they belong to. Lower is a "
                             "more responsive picture; see the note in the "
                             "configuration file")
    parser.add_argument("--no-tls", action="store_true",
                        help="serve plain HTTP; the browser microphone needs "
                             "a secure context and will not work")
    parser.add_argument("--capture-selection",
                        help="run.sh's USB camera selection file; the page shows the "
                             "capture mode the running pipeline selected")
    args = parser.parse_args(argv)

    config = load_ui_config(args.config)
    if args.port:
        config["web"]["port"] = args.port
    if args.bind:
        config["web"]["bind"] = args.bind
    if args.insight_host:
        config["insight"]["host"] = args.insight_host
        resolve_insight_bases(config, rebuild=True)
    if args.voice_host:
        config["voice"]["host"] = args.voice_host
    if args.voice_port:
        config["voice"]["port"] = args.voice_port
    if args.sync_buffer_ms is not None:
        if args.sync_buffer_ms < 0:
            parser.error("--sync-buffer-ms cannot be negative")
        config["render"]["video_sync_buffer_ms"] = args.sync_buffer_ms
    if args.no_tls:
        config["web"].setdefault("tls", {})["enabled"] = False

    # Insight was expected here and is not here: ask the machine where the SDK
    # host is and verify the answer, rather than serving a page with two empty
    # panels and a flag in the documentation.  Skipped entirely when the host
    # was stated, which is the normal case once it is pinned.
    if (not args.insight_host and not args.no_insight_autodetect
            and config["insight"]["host"] == "127.0.0.1"):
        local = insight_client.InsightClient(config["insight"]["api_base"],
                                             config["insight"]["vf_base"],
                                             timeout=2.5)
        if not local.identifies_as_insight():
            found = discover_insight_host(config)
            if found:
                config["insight"]["host"] = found
                config["insight"]["autodetected"] = True
                resolve_insight_bases(config, rebuild=True)
                log("using Insight at %s - pin it with \"host\": \"%s\" under "
                    "\"insight\" in %s" % (found, found, args.config))
            else:
                log("no Insight found on this network; pass --insight-host")

    app = App(config, capture_selection=args.capture_selection)
    Handler.app = app

    tls = resolve_tls(config)
    port = int(config["web"]["port"])
    bind = config["web"]["bind"]

    try:
        httpd = ThreadingHTTPServer((bind, port), Handler)
    except OSError as exc:
        import errno
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                "port %d on %s is already in use.\n"
                "  Something is holding it - very often an earlier copy of this\n"
                "  server whose pid file was lost. Check and stop it with:\n"
                "      %s/run.sh --status\n"
                "      %s/run.sh --stop\n"
                "  Or pick another port:  ./run.sh --port %d"
                % (port, bind, APP_DIR, APP_DIR, port + 1))
        raise
    httpd.daemon_threads = True
    scheme = "http"
    tls_note = ("Plain HTTP: the browser microphone is unavailable "
                "(getUserMedia needs HTTPS). Typed commands still work.")
    if tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(tls[0], tls[1])
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
        tls_note = "HTTPS with %s - the browser microphone works." % tls[0]

    # Issue 0: a bare SIGTERM would kill this process without
    # app.stop_background() ever running, and run.sh's Ctrl+C path sends one.
    #
    # The handler asks the server loop to stop rather than raising: on a real
    # terminal the tty delivers SIGINT to the whole foreground group, so this
    # process gets Ctrl+C AND then run.sh's SIGTERM a moment later, which is
    # during interpreter shutdown.  A handler that raises there prints a
    # KeyboardInterrupt traceback out of threading._shutdown for a shutdown
    # that actually succeeded.  Setting a flag and stopping the loop once is
    # idempotent, so the second signal is a no-op.
    stopping = threading.Event()

    def _terminate(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        log("signal %d - stopping" % signum)
        # serve_forever() cannot be stopped from inside its own handler, so the
        # request comes from a thread.  It returns as soon as the poll expires.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)

    app.start_background()
    banner(scheme, port, app, tls_note)
    log("listening on %s://%s:%d" % (scheme, bind, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:                      # a signal that beat the flag
        log("stopping")
    finally:
        # Anything arriving from here on is a duplicate of a shutdown already
        # in progress, and must not interrupt it.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        app.stop_background()
        httpd.server_close()
    log("stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(serve())
