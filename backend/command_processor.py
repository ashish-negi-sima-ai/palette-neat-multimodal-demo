#!/usr/bin/env python3
"""
command_processor.py - the ONE command path, shared by voice and typed text.

    python3 backend/command_processor.py --selftest
    python3 backend/command_processor.py "왼쪽 박스 없애줘"        (parser only, no Qwen3)

    voice:  microphone -> Whisper-medium (MLA) -> recognised text -+
    text:   text input ------------------------------------------ +-> process(text)

    process(text):
      1. deterministic parser (command_normalizer.py, software logic)
           complete and valid          -> execute it. Qwen3 is not used.
      2. otherwise (unknown / unsupported / incomplete / partial / ambiguous)
           the ORIGINAL text -> Qwen3-0.6B (MLA) -> strict JSON only
           -> schema check -> grounding check against the text
           valid                       -> execute it, shown with "(Qwen3)"
           anything else               -> "Command not understood", nothing runs

Both voice/board_voice_server.py (which owns Qwen3) and backend/server.py (when the
voice runtime is not reachable, parser only) call process(), so a given text always
produces the same command no matter how it arrived.

WHAT "VALIDATED" MEANS FOR A QWEN3 ANSWER

  strict output   the reply, after an empty <think></think> template block is
                  removed, must be exactly one JSON object: no prose, no Markdown,
                  no code fence.
  schema          only the keys action / camera / classes / color; a supported
                  action; every field that action needs, valid (camera left /
                  right / both, classes real model classes, colour a known
                  colour); no non-empty field the action does not take.
  grounding       the answer may not contain what the text does not say:
                    camera   named in the text (never guessed)
                    classes  exactly the classes named in the text
                    colour   the colour named in the text
                  and it may not drop what the text does say (a class or a
                  colour in the text that the action ignores is a partial
                  command). Camera/display ON-OFF never becomes a detection
                  toggle.
  class list      when the parser found an unresolved class term in a class-list
                  context ("사람 펀만": 펀 maps to no supported class through any
                  deterministic rule), NO answer is accepted. Qwen3 may not drop
                  the term ("PERSON ONLY"), guess a class for it, or turn the
                  request into another action. Only an existing deterministic
                  alias (컷 -> cup) resolves a term, and that happens in the parser.

A refused answer changes nothing: no partial command is executed and no missing
field is filled in.
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import command_config                                            # noqa: E402
import command_normalizer                                        # noqa: E402

ALLOWED_KEYS = ("action", "camera", "classes", "color")
QWEN_ATTEMPTS = 2
NOT_UNDERSTOOD = "Command not understood"

_EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*", re.I)


def strict_json_object(raw):
    """(dict, None) when `raw` is exactly one JSON object, else (None, why)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty output"
    text = _EMPTY_THINK.sub("", raw).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None, "output is not a bare JSON object: %r" % raw[:80]
    try:
        obj = json.loads(text)
    except ValueError as exc:
        return None, "malformed JSON (%s)" % exc
    if not isinstance(obj, dict):
        return None, "JSON is not an object"
    return obj, None


def _empty(value):
    return value is None or value == "" or value == []


def validate_model_command(obj, evidence, config):
    """(command, None) for a valid, grounded model answer; (None, why) otherwise."""
    extra = [k for k in obj if k not in ALLOWED_KEYS]
    if extra:
        return None, "unexpected key(s) %s" % extra
    action = obj.get("action")
    if not isinstance(action, str) or action not in command_config.SUPPORTED_ACTIONS:
        return None, "unsupported action %r" % (action,)
    if action == "unknown":
        return None, "Qwen3 answered unknown"
    fields = command_config.SUPPORTED_ACTIONS[action]
    for key in ALLOWED_KEYS[1:]:
        if key not in fields and not _empty(obj.get(key)):
            return None, "%s does not take %s=%r" % (action, key, obj.get(key))

    command = {"action": action}
    if "camera" in fields:
        camera = obj.get("camera")
        if camera not in command_config.CAMERAS:
            return None, "invalid camera %r" % (camera,)
        command["camera"] = camera
    if "classes" in fields:
        raw_classes = obj.get("classes")
        if isinstance(raw_classes, str):
            raw_classes = [raw_classes]
        if not isinstance(raw_classes, list) or not raw_classes:
            return None, "detect_only needs a non-empty class list"
        classes = []
        for value in raw_classes:
            name = config.matchers["object"].canonicalize(value)
            if name is None or name not in config.canonical_objects:
                return None, "not a model class: %r" % (value,)
            if name not in classes:
                classes.append(name)
        command["classes"] = classes
    if "color" in fields:
        color = config.matchers["color"].canonicalize(obj.get("color"))
        if color is None:
            return None, "not a supported colour: %r" % (obj.get("color"),)
        command["color"] = color

    # -- grounding --------------------------------------------------------
    ev = evidence
    unresolved = getattr(ev, "unresolved_class_terms", [])
    if unresolved:
        return None, ("unresolved class term(s) %s in the requested class list: not a "
                      "supported class by any deterministic rule, so the answer is not "
                      "accepted (a list item is never dropped or guessed)"
                      % ", ".join(unresolved))
    if action != "swap_camera":
        named = set(ev.cameras)
        if not named:
            return None, "the text names no camera (a camera is never guessed)"
        camera = command["camera"]
        if camera == "both":
            if not ("both" in named or {"left", "right"} <= named):
                return None, "answer says both cameras, the text says %s" % sorted(named)
        elif camera not in named or "both" in named or {"left", "right"} <= named:
            return None, "answer says %s, the text says %s" % (camera, sorted(named))
    elif not (ev.intents.get("swap") or ev.intents.get("display_target")
              or len(ev.cameras) >= 2):
        return None, "a swap for a text that is not about the cameras"

    if action == "detect_only":
        if set(command["classes"]) != set(ev.classes):
            return None, ("answer classes %s, the text names %s"
                          % (command["classes"], ev.classes))
        command["classes"] = [c for c in ev.classes]
    elif ev.classes:
        return None, "the text names %s but %s ignores them" % (ev.classes, action)

    if action == "set_box_color":
        if ev.color != command["color"]:
            return None, "answer colour %s, the text says %s" % (command["color"],
                                                                 ev.colors or "none")
    elif ev.colors:
        return None, "the text names colour %s but %s ignores it" % (ev.colors, action)

    if action in ("detection_on", "detection_off"):
        if ev.intents.get("display_target") and not ev.intents.get("detection_target"):
            return None, "camera/display ON-OFF is not supported"
        if (action == "detection_on" and ev.enabled == "false") or \
                (action == "detection_off" and ev.enabled == "true"):
            return None, "answer %s contradicts the text's ON/OFF word" % action

    problem = command_normalizer.check_command(command, config)
    if problem:
        return None, problem
    return command_normalizer.canonical_order(command), None


def describe(command, qwen=False):
    """The one-line text the UI shows for a command."""
    if not isinstance(command, dict) or command.get("action") in (None, "unknown"):
        return NOT_UNDERSTOOD
    action = command["action"]
    camera = str(command.get("camera", "")).upper()
    if action == "detect_only":
        classes = [c.upper() for c in command.get("classes", [])]
        body = "%s ONLY" % classes[0] if len(classes) == 1 else " + ".join(classes)
    elif action == "detect_all":
        body = "ALL CLASSES"
    elif action == "detection_on":
        body = "DETECTION ON"
    elif action == "detection_off":
        body = "DETECTION OFF"
    elif action == "reset":
        body = "RESET"
    elif action == "set_box_color":
        body = "BOX %s" % str(command.get("color", "")).upper()
    elif action == "swap_camera":
        camera, body = "SWAP", "LEFT ↔ RIGHT"
    else:
        body = action.upper()
    text = "%s / %s" % (camera, body)
    return text + (" (Qwen3)" if qwen else "")


def _result(text, status, command, source, reason, parsed, qwen_info, t0):
    qwen_used = source == "qwen3"
    return {
        "text": text,
        "status": status,
        "command": command,
        "source": source,
        "qwen_invoked": qwen_info is not None,
        "qwen_used": qwen_used,
        "qwen_raw": (qwen_info or {}).get("raw", "")[:400],
        "qwen_attempts": (qwen_info or {}).get("attempts", 0),
        "qwen_metrics": (qwen_info or {}).get("metrics"),
        "reason": reason,
        "parser": parsed.as_dict(),
        "display": describe(command, qwen_used) if status == "ok" else NOT_UNDERSTOOD,
        "timing_ms": {"parser": parsed_ms(parsed),
                      "qwen": (qwen_info or {}).get("ms", 0),
                      "total": int((time.time() - t0) * 1000)},
    }


def parsed_ms(parsed):
    return getattr(parsed, "elapsed_ms", 0)


def process(text, normalizer, qwen=None):
    """Parser first; Qwen3 only when the parser cannot decide.

    `qwen(text)` returns (raw_output, metrics_or_None) and runs Qwen3-0.6B on the
    MLA; None means the fallback is unavailable (the result is then simply not
    understood).
    """
    t0 = time.time()
    text = (text or "").strip()
    parsed = normalizer.parse(text)
    parsed.elapsed_ms = int((time.time() - t0) * 1000)
    if parsed.ok:
        return _result(text, "ok", parsed.command, "parser", None, parsed, None, t0)

    unknown = {"action": "unknown", "original_text": text}
    if not text:
        return _result(text, "not_understood", unknown, None, "empty text", parsed, None, t0)
    if qwen is None:
        return _result(text, "not_understood", unknown, None,
                       "parser: %s; Qwen3 fallback unavailable" % parsed.reason,
                       parsed, None, t0)

    info = {"attempts": 0, "raw": "", "metrics": None, "ms": 0}
    tq = time.time()
    obj, why = None, None
    for _ in range(QWEN_ATTEMPTS):
        info["attempts"] += 1
        try:
            raw, metrics = qwen(text)
        except Exception as exc:                                # noqa: BLE001
            why = "Qwen3 inference failed: %s" % exc
            break
        info["raw"], info["metrics"] = raw or "", metrics
        obj, why = strict_json_object(raw)
        if obj is not None:
            break
    info["ms"] = int((time.time() - tq) * 1000)
    if obj is None:
        return _result(text, "not_understood", unknown, None,
                       "parser: %s; Qwen3: %s" % (parsed.reason, why), parsed, info, t0)
    command, why = validate_model_command(obj, parsed.evidence, normalizer.config)
    if command is None:
        return _result(text, "not_understood", unknown, None,
                       "parser: %s; Qwen3 answer refused: %s" % (parsed.reason, why),
                       parsed, info, t0)
    return _result(text, "ok", command, "qwen3", None, parsed, info, t0)


# --------------------------------------------------------------------------
# Self-test (no model: Qwen3 answers are scripted)
# --------------------------------------------------------------------------

def _selftest():
    config = command_config.load()
    normalizer = command_normalizer.CommandNormalizer(config)
    failures = []

    def check(label, cond, extra=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                               "" if cond else "\n         -> " + str(extra)))
        if not cond:
            failures.append(label)

    def scripted(*answers):
        calls = []

        def qwen(text):
            calls.append(text)
            answer = answers[min(len(calls), len(answers)) - 1]
            return (answer if isinstance(answer, str)
                    else json.dumps(answer, ensure_ascii=False)), None
        qwen.calls = calls
        return qwen

    print("command_processor.py self-test\n-- parser first")
    q = scripted({"action": "detection_off", "camera": "left"})
    r = process("왼쪽 사람만 보여줘", normalizer, q)
    check("a complete parser command never calls Qwen3",
          r["source"] == "parser" and not q.calls and r["display"] == "LEFT / PERSON ONLY",
          r)
    r = process("왼쪽 카메라에서 사람하고 컵만 탐지해 주세요", normalizer, q)
    check("multi-class by the parser shows no Qwen3 marker",
          r["display"] == "LEFT / PERSON + CUP" and not q.calls, r["display"])

    print("-- Qwen3 fallback, valid answers")
    for text, answer, expect in (
            ("왼쪽 박스 없애줘", {"action": "detection_off", "camera": "left"},
             "LEFT / DETECTION OFF (Qwen3)"),
            ("Hide the boxes on the right", {"action": "detection_off", "camera": "right"},
             "RIGHT / DETECTION OFF (Qwen3)"),
            ("왼쪽 사람이랑 컵만 켜줘",
             {"action": "detect_only", "camera": "left", "classes": ["person", "cup"]},
             "LEFT / PERSON + CUP (Qwen3)"),
            ("왼쪽 사람이랑 컵만 켜줘",
             '<think>\n\n</think>\n\n{"action":"detect_only","camera":"left","classes":["people","cups"]}',
             "LEFT / PERSON + CUP (Qwen3)"),
    ):
        q = scripted(answer)
        r = process(text, normalizer, q)
        check("%r -> %s" % (text, expect),
              r["status"] == "ok" and r["display"] == expect and q.calls == [text], r)

    print("-- Qwen3 fallback, refused answers change nothing")
    for text, answers, why in (
            ("왼쪽 박스 없애줘", ['```json\n{"action":"detection_off","camera":"left"}\n```'] * 2,
             "Markdown fence"),
            ("왼쪽 박스 없애줘", ['Sure! {"action":"detection_off","camera":"left"}'] * 2,
             "prose"),
            ("왼쪽 박스 없애줘", ['{"action":"detection_off","camera":"left"'] * 2,
             "malformed"),
            ("왼쪽 박스 없애줘", [{"action": "detection_off", "camera": "right"}],
             "wrong camera"),
            ("박스 없애줘", [{"action": "detection_off", "camera": "left"}],
             "camera guessed"),
            ("왼쪽 박스 없애줘", [{"action": "detection_off", "camera": "left",
                                "reason": "user wants"}], "extra key"),
            ("왼쪽 사람 탐지 꺼", [{"action": "detection_off", "camera": "left"}],
             "drops the class (partial)"),
            ("왼쪽 사람이랑 컵만 켜줘",
             [{"action": "detect_only", "camera": "left", "classes": ["person"]}],
             "drops a class (partial)"),
            ("왼쪽 사람만 켜줘",
             [{"action": "detect_only", "camera": "left", "classes": ["person", "dog"]}],
             "invented class"),
            ("왼쪽 사람만 켜줘",
             [{"action": "detect_only", "camera": "left", "classes": ["wolf"]}],
             "not a model class"),
            ("오른쪽 카메라 꺼", [{"action": "detection_off", "camera": "right"}],
             "display OFF is not detection OFF"),
            ("왼쪽 박스 없애줘", [{"action": "display", "camera": "left"}],
             "unsupported action"),
            ("왼쪽 박스 없애줘", [{"action": "unknown"}], "unknown"),
            ("오늘 날씨 어때", [{"action": "swap_camera"}], "swap about nothing"),
            ("왼쪽 박스 없애줘", [{"action": "set_box_color", "camera": "left",
                                "color": "green"}], "invented colour"),
    ):
        q = scripted(*answers)
        r = process(text, normalizer, q)
        check("refused (%s): %r" % (why, text),
              r["status"] == "not_understood" and r["command"]["action"] == "unknown"
              and r["display"] == NOT_UNDERSTOOD and r["qwen_invoked"]
              and not r["qwen_used"], r)

    print("-- class-list guard: unresolved class terms (regressions)")
    q = scripted({"action": "detect_only", "camera": "left", "classes": ["person"]})
    r = process("왼쪽 카메라에서 사람하고 컵만 탐지해 주세요", normalizer, q)
    check("1. person + valid cup -> LEFT / PERSON + CUP (parser)",
          r["status"] == "ok" and r["display"] == "LEFT / PERSON + CUP" and not q.calls, r)
    for answer, why in (({"action": "detect_only", "camera": "left", "classes": ["person"]},
                         "Qwen3 drops the unknown item"),
                        ({"action": "detect_only", "camera": "left", "classes": ["person", "cup"]},
                         "Qwen3 guesses a class for it"),
                        ({"action": "detection_on", "camera": "left"}, "Qwen3 answers another action"),
                        ({"action": "detect_all", "camera": "left"}, "Qwen3 answers all classes")):
        q = scripted(answer)
        r = process("왼쪽 카메라 사람 펀만 탐지해 주세요", normalizer, q)
        check("2. person + unknown token 펀, %s -> Command not understood" % why,
              r["status"] == "not_understood" and r["display"] == NOT_UNDERSTOOD
              and r["command"]["action"] == "unknown" and r["qwen_invoked"] and not r["qwen_used"]
              and "펀" in (r["reason"] or "")
              and r["parser"]["evidence"]["unresolved_class_terms"] == ["펀"], r)
    q = scripted({"action": "detect_only", "camera": "left", "classes": ["person"]})
    r = process("왼쪽 카메라 사람 컷만 탐지해 주세요", normalizer, q)
    check("3. person + known Whisper alias 컷 -> LEFT / PERSON + CUP (parser)",
          r["status"] == "ok" and r["display"] == "LEFT / PERSON + CUP" and not q.calls, r)
    for text, answer in (("왼쪽 사람이랑 컵만 켜줘",
                          {"action": "detect_only", "camera": "left", "classes": ["person"]}),
                         ("왼쪽 사람이랑 컵이랑 병만 켜줘",
                          {"action": "detect_only", "camera": "left", "classes": ["person", "cup"]}),
                         ("Show only people and blorps on the left",
                          {"action": "detect_only", "camera": "left", "classes": ["person"]})):
        q = scripted(answer)
        r = process(text, normalizer, q)
        check("4. Qwen3 returns a subset of the requested class list -> rejected: %r" % text,
              r["status"] == "not_understood" and r["display"] == NOT_UNDERSTOOD
              and r["qwen_invoked"] and not r["qwen_used"], r)
    q = scripted({"action": "detection_off", "camera": "left"})
    r = process("왼쪽 카메라에서 사람만 좀 보여주세요", normalizer, q)
    check("a filler word (좀) is not a class candidate -> LEFT / PERSON ONLY",
          r["status"] == "ok" and r["display"] == "LEFT / PERSON ONLY" and not q.calls, r)

    q = scripted("not json", {"action": "detection_off", "camera": "left"})
    r = process("왼쪽 박스 없애줘", normalizer, q)
    check("one retry when the first reply is not a JSON object",
          r["status"] == "ok" and len(q.calls) == 2, r)
    r = process("왼쪽 박스 없애줘", normalizer, None)
    check("no Qwen3 available -> not understood, nothing guessed",
          r["status"] == "not_understood" and not r["qwen_invoked"], r)

    print("-- describe")
    for command, expect in (({"action": "detect_only", "camera": "left",
                              "classes": ["person", "cup", "bottle"]},
                             "LEFT / PERSON + CUP + BOTTLE"),
                            ({"action": "reset", "camera": "right"}, "RIGHT / RESET"),
                            ({"action": "set_box_color", "camera": "both", "color": "red"},
                             "BOTH / BOX RED")):
        check("%s" % expect, describe(command) == expect, describe(command))

    if failures:
        print("\n%d check(s) FAILED" % len(failures))
        return 1
    print("\nall checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("text", nargs="*")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if not args.text:
        parser.error("give a text, or --selftest")
    normalizer = command_normalizer.CommandNormalizer(command_config.load())
    print(json.dumps(process(" ".join(args.text), normalizer, None),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
