#!/usr/bin/env python3
"""
command_acceptance.py - end-to-end command tests against the RUNNING demo UI.

    python3 scripts/command_acceptance.py --base https://<devkit>:8022 --out runtime/test/<run>/acceptance.json
    python3 scripts/command_acceptance.py --sections required,reset,qwen --voice

Every command goes through the UI server's own endpoints - the ones the page uses:

    POST /api/text/command      typed text          -> command path
    POST /api/voice/command     a WAV (the page's microphone upload) -> Whisper -> command path
    POST /api/command           manual controls (used here only to set preconditions)

and each result is checked twice: the command the server reports, and the panel
state it actually produced. Voice clips are testdata/commands/*.wav (TTS, see
scripts/make_tts_clips.py). A TTS clip is not a live microphone in a room.

Sections: required reset qwen unsupported multiclass language equivalence
"""

import argparse
import json
import os
import ssl
import sys
import time
from urllib import error as urlerror
from urllib import request as urlrequest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(APP_DIR, "testdata", "commands")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
NOT_UNDERSTOOD = "Command not understood"

RESULTS = []


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def _call(self, path, data=None, content_type="application/json", timeout=120):
        req = urlrequest.Request(self.base + path, data=data,
                                 method="GET" if data is None else "POST",
                                 headers={"Content-Type": content_type})
        try:
            with urlrequest.urlopen(req, timeout=timeout, context=CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            return json.loads(exc.read().decode("utf-8") or "{}")

    def state(self):
        return self._call("/api/state")["state"]

    def manual(self, command):
        return self._call("/api/command", json.dumps({"command": command}).encode())

    def text(self, text):
        t0 = time.time()
        reply = self._call("/api/text/command", json.dumps({"text": text}).encode("utf-8"))
        reply["wall_ms"] = int((time.time() - t0) * 1000)
        return reply

    def voice(self, clip, language):
        with open(os.path.join(CLIPS, clip), "rb") as fh:
            wav = fh.read()
        t0 = time.time()
        for _ in range(10):
            reply = self._call("/api/voice/command?language=%s" % language, wav, "audio/wav")
            if (reply.get("command") or {}).get("display", "").startswith("Busy"):
                time.sleep(1.0)
                continue
            break
        reply["wall_ms"] = int((time.time() - t0) * 1000)
        return reply

    def reset_all(self):
        state = self.state()
        if state["mapping"] != state["initial_mapping"]:
            self.manual({"action": "swap_camera"})
        self.manual({"action": "reset", "camera": "both"})


def core(position_state):
    return {k: position_state[k] for k in ("detection_enabled", "classes", "box_color")}


def effect_ok(expect, before, after):
    """Does the state change match the expected command?"""
    if expect is None:
        return (after["positions"] == before["positions"] and after["mapping"] == before["mapping"]), \
            "state unchanged"
    action = expect["action"]
    if action == "swap_camera":
        swapped = after["mapping"] == {"left": before["mapping"]["right"],
                                       "right": before["mapping"]["left"]}
        # Each camera view moves as one unit: the view now on the left is the one that
        # was on the right, with all of its overlay state.
        travelled = (core(after["positions"]["left"]) == core(before["positions"]["right"])
                     and core(after["positions"]["right"]) == core(before["positions"]["left"]))
        return swapped and travelled, "mapping swapped, views moved whole" if swapped and travelled \
            else "mapping %s, view state travelled %s" % (swapped, travelled)
    cams = ["left", "right"] if expect["camera"] == "both" else [expect["camera"]]
    for cam in cams:
        p = after["positions"][cam]
        if action == "detect_only" and not (p["classes"] == expect["classes"] and p["detection_enabled"]):
            return False, "classes %s on %s" % (p["classes"], cam)
        if action == "detection_off" and p["detection_enabled"] is not False:
            return False, "detection still on"
        if action == "detection_on" and p["detection_enabled"] is not True:
            return False, "detection still off"
        if action == "reset" and not p["is_default"]:
            return False, "not default: %s" % core(p)
        if action == "detect_all" and not (p["classes"] is None and p["detection_enabled"]):
            return False, "classes %s" % p["classes"]
        if action == "set_box_color" and p["box_color"] != expect["color"]:
            return False, "box colour %s" % p["box_color"]
    others = [c for c in ("left", "right") if c not in cams]
    for cam in others:
        if action != "swap_camera" and core(after["positions"][cam]) != core(before["positions"][cam]):
            return False, "the other camera changed"
    return True, "ok"


def record(section, label, path, text, reply, expect, before, after, want_qwen=None):
    cmd = reply.get("command") or {}
    got = cmd.get("command")
    if expect is None:
        command_ok = cmd.get("display") == NOT_UNDERSTOOD
    else:
        command_ok = got == expect and cmd.get("status") == "ok"
    eff, eff_note = effect_ok(expect, before, after)
    qwen_ok = True if want_qwen is None else (bool(cmd.get("qwen_used")) == want_qwen)
    display = cmd.get("display")
    marker_ok = ("(Qwen3)" in (display or "")) == bool(cmd.get("qwen_used"))
    passed = command_ok and eff and qwen_ok and marker_ok
    voice_reply = reply.get("voice_reply") or {}
    row = {"section": section, "label": label, "path": path, "text": text,
           "recognized": cmd.get("input"), "display": display, "command": got,
           "expect": expect, "source": cmd.get("source"), "qwen_used": cmd.get("qwen_used"),
           "qwen_invoked": cmd.get("qwen_invoked"), "reason": cmd.get("reason"),
           "effect": eff_note, "passed": passed, "wall_ms": reply.get("wall_ms"),
           "timing_ms": cmd.get("timing_ms"), "language": cmd.get("language"),
           "detected_language": voice_reply.get("detected_language"),
           "whisper_confidence": voice_reply.get("whisper_confidence")}
    RESULTS.append(row)
    print("  [%s] %-6s %-11s %-44s -> %-32s %s%s" % (
        "PASS" if passed else "FAIL", section[:6], path, (text or "")[:44], display,
        ("recognized=%r " % cmd.get("input")) if path.startswith("voice") else "",
        ("" if passed else "(%s; %s; reason=%s)" % (eff_note, got, cmd.get("reason")))))
    return row


def run_one(client, section, label, path, text_or_clip, expect, language="ko",
            precondition=None, want_qwen=None):
    client.reset_all()
    for command in precondition or []:
        client.manual(command)
    if expect and expect["action"] == "reset":
        cams = ["left", "right"] if expect["camera"] == "both" else [expect["camera"]]
        for cam in cams:
            client.manual({"action": "set_box_color", "camera": cam, "color": "green"})
            client.manual({"action": "detect_only", "camera": cam, "classes": ["person"]})
            client.manual({"action": "detection_off", "camera": cam})
    before = client.state()
    if path == "text":
        reply = client.text(text_or_clip)
        text = text_or_clip
    else:
        reply = client.voice(text_or_clip["file"], language)
        text = text_or_clip["text"]
    after = client.state()
    return record(section, label, path if path == "text" else "voice-%s" % language, text,
                  reply, expect, before, after, want_qwen)


def manifest():
    return json.load(open(os.path.join(CLIPS, "manifest.json"), encoding="utf-8"))


NON_DEFAULT = [{"action": "detect_only", "camera": "left", "classes": ["person"]},
               {"action": "set_box_color", "camera": "left", "color": "red"},
               {"action": "detection_off", "camera": "right"}]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--base", default="https://10.42.0.203:8022")
    ap.add_argument("--out")
    ap.add_argument("--sections", default="required,reset,qwen,unsupported,multiclass,language,equivalence")
    ap.add_argument("--voice", action="store_true", help="also send the voice clips")
    args = ap.parse_args()
    client = Client(args.base)
    sections = args.sections.split(",")
    clips = manifest()
    started = time.time()

    if "required" in sections:
        print("\n== required commands (text%s)" % (" and voice" if args.voice else ""))
        for e in [c for c in clips if c["group"] in ("required", "extra")]:
            run_one(client, "required", e["file"], "text", e["text"], e["expect"])
            if args.voice:
                run_one(client, "required", e["file"], "voice", e, e["expect"], e["lang"])

    if "reset" in sections:
        print("\n== reset: green box + detection off + class filter, then reset")
        for side, text in (("left", "왼쪽 카메라 초기화하세요"), ("right", "오른쪽 카메라 초기화하세요"),
                           ("left", "Reset the left camera"), ("right", "Reset the right camera")):
            run_one(client, "reset", side, "text", text, {"action": "reset", "camera": side})
        if args.voice:
            for e in [c for c in clips if c["group"] == "reset"] + \
                    [c for c in clips if c["file"] == "en_left_reset.wav"]:
                run_one(client, "reset", e["file"], "voice", e, e["expect"], e["lang"])

    if "qwen" in sections:
        print("\n== Qwen3 fallback (the parser does not decide these)")
        for text, expect in (("Hide the boxes on the right", {"action": "detection_off", "camera": "right"}),
                             ("Remove the boxes from the left camera", {"action": "detection_off", "camera": "left"}),
                             ("오른쪽 카메라 기본 상태로 돌려줘", {"action": "reset", "camera": "right"}),
                             ("Make the right camera like it was at the start", {"action": "reset", "camera": "right"}),
                             ("왼쪽 사람이랑 컵만 켜줘", {"action": "detect_only", "camera": "left", "classes": ["person", "cup"]})):
            run_one(client, "qwen", text, "text", text, expect, want_qwen=True)
        if args.voice:
            for e in [c for c in clips if c["group"] == "qwen"]:
                run_one(client, "qwen", e["file"], "voice", e, e["expect"], e["lang"], want_qwen=True)

    if "unsupported" in sections:
        print("\n== not understood: nothing changes (non-default state first)")
        for text in ("오늘 날씨 어때", "오른쪽 카메라 꺼", "왼쪽 화면 박스 지워줘",
                     "왼쪽 카메라 사람 펀만 탐지해 주세요",
                     "사람이 보이면 오른쪽 카메라로 바꾸고 컵이 나오면 빨간 박스를 그려줘"):
            run_one(client, "unsupp", text, "text", text, None, precondition=NON_DEFAULT)
        if args.voice:
            for e in [c for c in clips if c["group"] == "unsupported"]:
                run_one(client, "unsupp", e["file"], "voice", e, None, e["lang"], precondition=NON_DEFAULT)

    if "multiclass" in sections:
        print("\n== multi-class")
        for text, classes in (("왼쪽 카메라 사람 컷만 탐지해 주세요", ["person", "cup"]),
                              ("왼쪽 카메라에서 사람만 좀 보여주세요", ["person"]),
                              ("왼쪽 사람하고 컵 보여줘", ["person", "cup"]),
                              ("왼쪽에서 person, cup, bottle만 보여줘", ["person", "cup", "bottle"]),
                              ("Show only people and cups on the left", ["person", "cup"])):
            run_one(client, "multi", text, "text", text,
                    {"action": "detect_only", "camera": "left", "classes": classes})
        if args.voice:
            for e in [c for c in clips if c["group"] == "multiclass"]:
                run_one(client, "multi", e["file"], "voice", e, e["expect"], e["lang"])

    if "language" in sections and args.voice:
        print("\n== language modes")
        for e in clips:
            if e["group"] not in ("required", "english") or e["expect"] is None:
                continue
            modes = ("ko", "auto") if e["lang"] == "ko" else ("en", "auto")
            for mode in modes:
                run_one(client, "lang", e["file"], "voice", e, e["expect"], mode)

    if "language" in sections and args.voice:
        print("\n== language stress: slow Korean, UK / Australian English, fixed and Auto")
        for e in [c for c in clips if c["group"] == "language"]:
            for mode in (e["lang"], "auto"):
                run_one(client, "lang2", e["file"], "voice", e, e["expect"], mode)

    if "equivalence" in sections and args.voice:
        print("\n== voice / text equivalence: the recognized text, typed, gives the same result")
        voice_rows = [r for r in RESULTS if r["path"].startswith("voice") and r["recognized"]]
        seen = set()
        for row in voice_rows:
            if row["recognized"] in seen:
                continue
            seen.add(row["recognized"])
            client.reset_all()
            if row["expect"] and row["expect"]["action"] == "reset":
                run_one(client, "equiv", row["label"], "text", row["recognized"], row["command"]
                        if row["command"] and row["command"].get("action") != "unknown" else None)
                RESULTS[-1]["voice_command"] = row["command"]
                RESULTS[-1]["same_as_voice"] = RESULTS[-1]["command"] == row["command"]
                continue
            reply = client.text(row["recognized"])
            text_cmd = (reply.get("command") or {}).get("command")
            same = text_cmd == row["command"] and \
                (reply.get("command") or {}).get("display") == row["display"]
            RESULTS.append({"section": "equiv", "label": row["label"], "path": "text",
                            "text": row["recognized"], "command": text_cmd,
                            "voice_command": row["command"], "display": (reply.get("command") or {}).get("display"),
                            "voice_display": row["display"], "passed": same, "same_as_voice": same})
            print("  [%s] equiv  %-44s voice=%s text=%s" % ("PASS" if same else "FAIL",
                                                           row["recognized"][:44], row["display"],
                                                           (reply.get("command") or {}).get("display")))
        client.reset_all()

    client.reset_all()
    passed = sum(1 for r in RESULTS if r["passed"])
    summary = {"base": args.base, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
               "duration_s": round(time.time() - started, 1), "passed": passed, "total": len(RESULTS)}
    print("\n%d / %d passed in %.0f s" % (passed, len(RESULTS), summary["duration_s"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": RESULTS}, fh, ensure_ascii=False, indent=1)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
