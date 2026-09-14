#!/usr/bin/env python3
"""
continuity_probe.py - is the video actually continuous?  Measured, not assumed.

    python3 scripts/continuity_probe.py --seconds 60                    # TEST A
    python3 scripts/continuity_probe.py --infer whisper --repeat 3      # TEST B
    python3 scripts/continuity_probe.py --infer qwen --no-pause         # TEST B'
    python3 scripts/continuity_probe.py --ui-only --repeat 5            # TEST C
    python3 scripts/continuity_probe.py --voice ko_05 --repeat 3        # TEST D

WHAT IT MEASURES, AND WHY THERE

It samples Neat Insight's OWN per-channel RTP packet counters several times a
second.  Those counters are incremented by vf as UDP arrives from the DevKit, so
they sit UPSTREAM of the browser, upstream of WebRTC and upstream of this
application entirely.  If they stop advancing, the board stopped sending: no
amount of browser or UI behaviour can cause that, and no browser or UI fix can
repair it.  If they keep advancing while the picture still breaks, the cause is
downstream instead.

That is the measurement that separates the three candidate causes:

    counters stall during voice     the VISION pipeline stopped sending
    counters fine, browser breaks   the UI / WebRTC lifecycle
    counters stall with no voice    the vision pipeline is unstable anyway

It also reads `forwarding.packets_forwarded` and `webrtc.peer_count`, so a
browser that is attached can be seen losing and regaining its track.

READ-ONLY.  Every request is a GET against Insight, except the action under
test.  Nothing is started, stopped, reconfigured or written.

THE ACTIONS

    (none)              just watch.  This is TEST A.
    --infer MODEL       POST nothing; GET the voice server's /debug/infer,
                        which runs ONE model in isolation and updates no UI at
                        all.  This is TEST B, and `--no-pause` runs it without
                        the vision pause so the two mechanisms can be told
                        apart.  MODEL is whisper | qwen | both | none.
    --ui-only           push a synthetic STT/QUERY/RESULT/STATUS update through
                        this demo's own query API with no audio and no models.
                        This is TEST C.
    --voice WAV         a real spoken command end to end.  This is TEST D.

`--infer` and `--voice` reach the voice server; everything else reaches only
this demo and Insight.
"""

import argparse
import json
import os
import ssl
import sys
import time
from urllib import error as urlerror
from urllib import request as urlrequest

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
TESTDATA = os.path.join(APP_DIR, "testdata")
# READ-ONLY: the recorded WAV corpus in testdata/reference.
REFERENCE_TESTDATA = os.environ.get(
    "REFERENCE_TESTDATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata", "reference"))
CONFIG_PATH = os.path.join(APP_DIR, "config", "ui_config.json")

_CONTEXT = ssl.create_default_context()
_CONTEXT.check_hostname = False
_CONTEXT.verify_mode = ssl.CERT_NONE

# A sample with fewer than this many new packets is a stall.  At 30 fps and
# ~370 packets/s a healthy 250 ms sample carries ~90; the tail of a stopping
# stream has been measured at 9.  The floor sits far below normal and far above
# noise.
STALL_FLOOR = 5


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def insight_base(config, override=None):
    insight = config["insight"]
    host = override or insight.get("host") or "127.0.0.1"
    if host in ("local", "localhost", ""):
        host = "127.0.0.1"
    bracketed = "[%s]" % host if ":" in host else host
    return "https://%s:%d" % (bracketed, int(insight.get("api_port", 9900)))


def get_json(url, timeout=8.0):
    with urlrequest.urlopen(url, timeout=timeout, context=_CONTEXT) as resp:
        return json.load(resp)


def post_json(url, payload, timeout=30.0):
    request = urlrequest.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urlrequest.urlopen(request, timeout=timeout, context=_CONTEXT) as resp:
        return json.load(resp)


def post_wav(url, wav, timeout=180.0):
    request = urlrequest.Request(url, data=wav,
                                 headers={"Content-Type": "audio/wav"},
                                 method="POST")
    try:
        with urlrequest.urlopen(request, timeout=timeout, context=_CONTEXT) as resp:
            return json.load(resp)
    except urlerror.HTTPError as exc:
        return {"error": exc.read().decode("utf-8", "replace")[:300]}


def sample(base, channels):
    """One reading of Insight's counters for the channels under test."""
    stats = get_json(base + "/api/ingest/stats")
    out = {}
    for entry in stats.get("channels", []):
        index = entry.get("channel")
        if index not in channels:
            continue
        rtp = entry.get("rtp") or {}
        fwd = entry.get("forwarding") or {}
        webrtc = entry.get("webrtc") or {}
        media = entry.get("media") or {}
        meta = entry.get("metadata") or {}
        out[index] = {
            "packets": rtp.get("packets_received"),
            "forwarded": fwd.get("packets_forwarded"),
            "dropped_no_track": fwd.get("packets_dropped_no_track"),
            "track": bool(fwd.get("webrtc_track_attached")),
            "peers": webrtc.get("peer_count"),
            "idr": media.get("idr_count"),
            "meta_msgs": meta.get("messages_received"),
            "bitrate": rtp.get("bitrate_bps"),
        }
    return out


class Trace:
    """The timeline, and the verdict drawn from it."""

    def __init__(self, channels):
        self.channels = channels
        self.rows = []
        self.events = []

    def add(self, t, previous, current):
        row = {"t": t, "channels": {}}
        for channel in self.channels:
            before = (previous or {}).get(channel) or {}
            after = current.get(channel) or {}
            delta = None
            if isinstance(before.get("packets"), int) \
                    and isinstance(after.get("packets"), int):
                delta = after["packets"] - before["packets"]
            fdelta = None
            if isinstance(before.get("forwarded"), int) \
                    and isinstance(after.get("forwarded"), int):
                fdelta = after["forwarded"] - before["forwarded"]
            row["channels"][channel] = {
                "packets_delta": delta,
                "forwarded_delta": fdelta,
                "idr_delta": (after.get("idr") or 0) - (before.get("idr") or 0)
                             if isinstance(before.get("idr"), int) else None,
                "meta_delta": (after.get("meta_msgs") or 0) - (before.get("meta_msgs") or 0)
                              if isinstance(before.get("meta_msgs"), int) else None,
                "track": after.get("track"),
                "peers": after.get("peers"),
            }
        self.rows.append(row)
        return row

    def note(self, t, text):
        self.events.append({"t": t, "text": text})

    def print_timeline(self, only_interesting=False):
        header = "   t(s)  " + "  ".join(
            "ch%d pkts/s  fwd/s" % c for c in self.channels) + "   event"
        print(header)
        print("   " + "-" * (len(header) - 3))
        events = list(self.events)
        for row in self.rows[1:]:
            text = ""
            while events and events[0]["t"] <= row["t"]:
                text = (text + " | " if text else "") + events.pop(0)["text"]
            stalled = any((row["channels"][c]["packets_delta"] or 0) <= STALL_FLOOR
                          for c in self.channels)
            if only_interesting and not stalled and not text:
                continue
            cells = []
            for channel in self.channels:
                entry = row["channels"][channel]
                cells.append("%9s  %5s" % (entry["packets_delta"],
                                           entry["forwarded_delta"]))
            mark = " STALL" if stalled else ""
            print("  %5.1f  %s   %s%s" % (row["t"], "  ".join(cells), text, mark))
        for event in events:
            print("  %5.1f  %s" % (event["t"], event["text"]))

    def verdict(self, window_s):
        """Per-channel: how many samples stalled, and the longest run."""
        out = {}
        for channel in self.channels:
            stalls = 0
            longest = 0
            run = 0
            total = 0
            for row in self.rows[1:]:
                delta = row["channels"][channel]["packets_delta"]
                if delta is None:
                    continue
                total += 1
                if delta <= STALL_FLOOR:
                    stalls += 1
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 0
            out[channel] = {
                "samples": total,
                "stalled_samples": stalls,
                "longest_stall_s": round(longest * window_s, 2),
                "stalled_s": round(stalls * window_s, 2),
            }
        return out


def run_action(args, config, trace, t0):
    """Whatever this test is supposed to do while the counters are watched."""
    voice = config["voice"]
    voice_base = "http://%s:%d" % (args.voice_host or voice["host"],
                                   args.voice_port or voice["port"])
    ui_base = args.base

    def now():
        return time.time() - t0

    if args.infer:
        pause = "0" if args.no_pause else "1"
        url = "%s/debug/infer?model=%s&pause=%s" % (voice_base, args.infer, pause)
        trace.note(now(), "INFER %s pause=%s start" % (args.infer, pause))
        try:
            reply = get_json(url, timeout=180)
            trace.note(now(), "INFER %s done %sms" % (args.infer,
                                                      (reply.get("timing_ms") or {}).get("total")))
            return reply
        except Exception as exc:                                # noqa: BLE001
            trace.note(now(), "INFER %s FAILED: %s" % (args.infer, exc))
            return {"error": str(exc)}

    if args.ui_only:
        # This project's OWN typed-transcript path: the real normalizer, the
        # real handler, a real state change and a real STT / COMMAND / VOICE AI
        # update - with no audio, no Whisper and no Qwen.  That is TEST C.
        text = args.ui_text or "오른쪽에서 사람만 보여주세요"
        trace.note(now(), "UI command start (no models): %s" % text)
        try:
            reply = post_json(ui_base + "/api/transcript", {"text": text})
            command = (reply.get("command") or {}).get("final")
            trace.note(now(), "UI command done: %s"
                       % json.dumps(command, ensure_ascii=False))
            return reply
        except Exception as exc:                                # noqa: BLE001
            trace.note(now(), "UI command FAILED: %s" % exc)
            return {"error": str(exc)}

    if args.voice:
        # A bare name resolves against this project's testdata/, then against
        # the read-only recorded corpus in testdata/reference
        # (and its tts/ subdirectory).  Nothing is ever copied or written there.
        candidates = [args.voice,
                      os.path.join(TESTDATA, "%s.wav" % args.voice),
                      os.path.join(REFERENCE_TESTDATA, "%s.wav" % args.voice),
                      os.path.join(REFERENCE_TESTDATA, "tts", "%s.wav" % args.voice)]
        path = next((c for c in candidates if os.path.isfile(c)), candidates[1])
        if not os.path.isfile(path):
            trace.note(now(), "VOICE %s: no such clip" % args.voice)
            return {"error": "missing clip"}
        with open(path, "rb") as handle:
            wav = handle.read()
        trace.note(now(), "VOICE %s start" % args.voice)
        reply = post_wav(ui_base + "/api/voice/command?language=ko", wav)
        trace.note(now(), "VOICE %s done" % args.voice)
        return reply

    return None



def default_base():
    """The UI's own configured port (web.port), not a hard-coded one."""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "config", "ui_config.json"),
                encoding="utf-8") as handle:
            return "https://127.0.0.1:%d" % int(
                json.load(handle)["web"]["port"])
    except Exception:                                            # noqa: BLE001
        return "https://127.0.0.1:8022"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--base", default=default_base(),
                        help="this demo's own server")
    parser.add_argument("--insight-host", default=None)
    parser.add_argument("--voice-host", default=None)
    parser.add_argument("--voice-port", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=0,
                        help="watch for this long with no action (TEST A)")
    parser.add_argument("--window", type=float, default=0.25,
                        help="sampling interval (default: %(default)s)")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="seconds of baseline before and after each action")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gap", type=float, default=4.0,
                        help="seconds between repeats")
    parser.add_argument("--infer", choices=["whisper", "qwen", "both", "none"],
                        help="run ONE model in isolation, no UI update (TEST B)")
    parser.add_argument("--no-pause", action="store_true",
                        help="with --infer: do NOT take the vision pause lease")
    parser.add_argument("--ui-only", action="store_true",
                        help="this project's typed-transcript path: a real "
                             "command applied with no models (TEST C)")
    parser.add_argument("--ui-text", default=None,
                        help="the transcript --ui-only sends")
    parser.add_argument("--voice", help="a real spoken question (TEST D)")
    parser.add_argument("--all-rows", action="store_true",
                        help="print every sample, not only the interesting ones")
    parser.add_argument("--json", help="write the raw trace here")
    args = parser.parse_args()

    config = load_config()
    base = insight_base(config, args.insight_host)
    channels = sorted(int(entry["channel"])
                      for entry in config["sources"].values())

    try:
        health = get_json(base + "/api/health")
    except Exception as exc:                                    # noqa: BLE001
        print("cannot reach Insight at %s: %s" % (base, exc), file=sys.stderr)
        return 2
    if health.get("service") != "neat-insight":
        print("%s is not Insight: %s" % (base, health), file=sys.stderr)
        return 2

    label = ("TEST A  vision only" if not (args.infer or args.ui_only or args.voice)
             else "TEST B  %s inference, no UI update%s"
                  % (args.infer, ", NO vision pause" if args.no_pause else "")
                  if args.infer
             else "TEST C  UI text update, no models" if args.ui_only
             else "TEST D  full voice query: %s" % args.voice)
    print("%s" % label)
    print("Insight   : %s   channels %s" % (base, channels))
    print("sampling  : every %.2f s   stall floor: <=%d packets per sample"
          % (args.window, STALL_FLOOR))
    print("")

    trace = Trace(channels)
    t0 = time.time()
    previous = None

    def tick():
        nonlocal previous
        current = sample(base, channels)
        row = trace.add(time.time() - t0, previous, current)
        previous = current
        return row

    def watch(seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            tick()
            time.sleep(args.window)

    tick()                                   # the first reading is the baseline
    # ...and one full window must pass before the next, or the first delta is
    # measured over a few milliseconds and reads as a stall that never was.
    time.sleep(args.window)
    if args.seconds:
        watch(args.seconds)
    else:
        for index in range(args.repeat):
            watch(args.settle)
            reply = run_action(args, config, trace, t0)
            if reply and reply.get("error"):
                print("  action error: %s" % reply["error"])
            watch(args.settle + 6.0)         # long enough to see a restore
            if index + 1 < args.repeat:
                watch(args.gap)

    trace.print_timeline(only_interesting=not args.all_rows)
    print("")
    verdict = trace.verdict(args.window)
    worst = 0.0
    for channel in channels:
        entry = verdict[channel]
        print("  channel %d : %d samples, %d stalled, longest stall %.2f s, "
              "total stalled %.2f s"
              % (channel, entry["samples"], entry["stalled_samples"],
                 entry["longest_stall_s"], entry["stalled_s"]))
        worst = max(worst, entry["longest_stall_s"])
    print("")
    # One sample of slack: a 250 ms window can straddle a scheduling hiccup.
    ok = worst <= args.window * 1.5
    print("  VERDICT: %s   (longest stall %.2f s)"
          % ("PASS - the board never stopped sending" if ok
             else "FAIL - the board STOPPED SENDING", worst))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"label": label, "window_s": args.window,
                       "rows": trace.rows, "events": trace.events,
                       "verdict": verdict}, handle, ensure_ascii=False, indent=1)
        print("  raw trace: %s" % args.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
