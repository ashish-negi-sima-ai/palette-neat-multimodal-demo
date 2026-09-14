#!/usr/bin/env python3
"""
voice_command_soak.py - repeated spoken commands through the REAL demo path.

    python3 scripts/voice_command_soak.py --base https://127.0.0.1:8022 \
        --clips ko_02,ko_05,ko_01,unk_ko --repeat 5 --gap 20 --out runtime/devkit/f/soak.jsonl

Each clip (a recorded 16 kHz WAV from testdata/reference, standing in for the
browser microphone) is POSTed to the UI server's /api/voice/command, exactly as
web/app.js does. The UI server forwards it to the on-device voice runtime, where
the steps are:

1. Whisper-medium transcribes it on the MLA.
2. The deterministic parser decides the command if it can.
3. Only when the parser cannot decide, Qwen3-0.6B runs on the MLA.
4. The UI server applies the final command.

Recorded per command:

- transcript;
- parser decision;
- whether Qwen was invoked, with its raw output and metrics (TTFT, tokens/s);
- the final applied command;
- the voice runtime's timing;
- the end-to-end latency seen by the caller.

This script only sends audio and reads the replies. It runs no model itself.
"""

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.request

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post_wav(base, wav_bytes, language, timeout):
    ctx = ssl._create_unverified_context()  # the demo UI uses a self-signed certificate
    req = urllib.request.Request(base.rstrip("/") + "/api/voice/command?language=" + language,
                                 data=wav_bytes, headers={"Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--base", default="https://127.0.0.1:8022")
    ap.add_argument("--clips", default="ko_02,ko_05,ko_01,unk_ko")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--gap", type=float, default=20.0, help="seconds between commands")
    ap.add_argument("--language", default="ko")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    clips = [c for c in args.clips.split(",") if c]
    rows = []
    with open(args.out, "a", buffering=1) as out:
        for rep in range(args.repeat):
            for clip in clips:
                wav = open(os.path.join(APP, "testdata", "reference", clip + ".wav"), "rb").read()
                t0 = time.time()
                rec = {"t": round(t0, 3), "rep": rep, "clip": clip}
                try:
                    reply = post_wav(args.base, wav, args.language, args.timeout)
                    rec["e2e_ms"] = int((time.time() - t0) * 1000)
                    vr = reply.get("voice_reply") or {}
                    cmd = reply.get("command") or {}
                    rec.update({
                        "ok": True,
                        "transcript": vr.get("stt_text"),
                        "decision": vr.get("decision"),
                        "qwen_invoked": vr.get("qwen_invoked"),
                        "qwen_raw": vr.get("qwen_raw"),
                        "qwen_metrics": vr.get("qwen_metrics"),
                        "vision_lease": vr.get("vision_lease"),
                        "voice_timing_ms": vr.get("timing_ms"),
                        "voice_command": vr.get("command"),
                        "final": cmd.get("final") if isinstance(cmd, dict) else cmd,
                        "status": cmd.get("status") if isinstance(cmd, dict) else None,
                    })
                except Exception as exc:  # noqa: BLE001
                    rec.update({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                                "e2e_ms": int((time.time() - t0) * 1000)})
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append(rec)
                print("%s %-7s e2e %5s ms qwen=%-5s %-30r -> %s" % (
                    time.strftime("%H:%M:%S", time.localtime(t0)), clip, rec.get("e2e_ms"),
                    rec.get("qwen_invoked"), rec.get("transcript"),
                    json.dumps(rec.get("final") or rec.get("error"), ensure_ascii=False)[:120]),
                    flush=True)
                time.sleep(args.gap)
    good = [r for r in rows if r.get("ok")]
    with_q = [r for r in good if r.get("qwen_invoked")]
    print("commands %d ok %d errors %d | qwen invoked %d, parser-only %d" % (
        len(rows), len(good), len(rows) - len(good), len(with_q), len(good) - len(with_q)))
    if good:
        e2e = [r["e2e_ms"] for r in good]
        print("e2e ms min %d median %d max %d" % (min(e2e), statistics.median(e2e), max(e2e)))
    return 0 if len(good) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
