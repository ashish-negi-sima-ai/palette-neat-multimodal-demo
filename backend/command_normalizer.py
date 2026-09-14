#!/usr/bin/env python3
"""
command_normalizer.py - the deterministic command parser.

    python3 backend/command_normalizer.py --selftest
    python3 backend/command_normalizer.py "왼쪽 카메라에서 사람하고 컵만 탐지해 주세요"

This is ordinary software logic, not an AI model and not an MLA workload. It turns
one transcript - spoken (after Whisper) or typed - into ONE canonical command, or
says why it cannot:

    ok            a complete, valid command with every required field resolved.
                  It is executed directly; Qwen3 is not used.
    unknown       nothing recognised, incomplete, ambiguous or partial.
    unsupported   positively identified and deliberately unavailable
                  (camera / display ON-OFF).

Anything that is not `ok` is handed, as the ORIGINAL text, to the Qwen3-0.6B
fallback in backend/command_processor.py. That module also uses the Evidence this
parser collects to validate the model's answer against what the text says.

THE CANONICAL COMMAND (the only thing backend/panel_state.py accepts)

    {"action": "detect_only",   "camera": C, "classes": ["person", "cup"]}
    {"action": "detect_all",    "camera": C}
    {"action": "detection_on",  "camera": C}
    {"action": "detection_off", "camera": C}
    {"action": "reset",         "camera": C}
    {"action": "set_box_color", "camera": C, "color": "green"}
    {"action": "swap_camera"}
    {"action": "unknown",       "original_text": "..."}

    C = left | right | both

Phrase matching (longest phrase wins, no fuzzy matching, no nearest-class guess) is
in command_config.py; the phrases themselves are in command_config.json.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import command_config                                            # noqa: E402

SUPPORTED_ACTIONS = command_config.SUPPORTED_ACTIONS
UNSUPPORTED_ACTIONS = command_config.UNSUPPORTED_ACTIONS

OK = "ok"
UNSUPPORTED = "unsupported"
UNKNOWN = "unknown"

INTENTS = ("show_only", "detect_verb", "detection_target", "display_target",
           "swap", "reset", "show_all", "color_target", "conditional")
# Intents whose phrases can also be part of a camera phrase ("좌우 모두" is
# `both`, "모두" alone is show-all), so they are read with camera phrases blanked.
CAMERA_MASKED_INTENTS = ("reset", "show_all")


class Evidence:
    """What one text literally says, slot by slot. Nothing here is inferred."""

    def __init__(self, text):
        self.text = text
        self.cameras = []          # distinct camera canonicals, text order
        self.camera = None         # resolved camera, or None
        self.camera_reason = None
        self.classes = []          # distinct object canonicals, text order
        self.colors = []
        self.color = None
        self.enabled = None        # "true" / "false" / None
        self.enabled_reason = None
        self.intents = {}
        # Words joined into a class list ("사람 펀만", "people and blorps") that no
        # deterministic rule maps to a supported class. Kept so that neither the parser
        # nor the Qwen3 fallback can run the command as a shorter list or guess a class.
        self.unresolved_class_terms = []

    def as_dict(self):
        return {"cameras": self.cameras, "camera": self.camera,
                "classes": self.classes, "colors": self.colors, "color": self.color,
                "enabled": self.enabled,
                "unresolved_class_terms": self.unresolved_class_terms,
                "intents": sorted(k for k, v in self.intents.items() if v)}


class Parsed:
    """One parser reading: status, the canonical command and why."""

    def __init__(self, status, command, reason, rule, evidence):
        self.status = status
        self.command = command
        self.reason = reason
        self.rule = rule
        self.evidence = evidence

    @property
    def ok(self):
        return self.status == OK

    def as_dict(self):
        return {"status": self.status, "command": self.command,
                "reason": self.reason, "rule": self.rule,
                "evidence": self.evidence.as_dict()}

    def __repr__(self):
        return "Parsed(%s, %s)" % (self.status,
                                   json.dumps(self.command, ensure_ascii=False))


def _overlaps(span, others):
    return any(span["start"] < o["end"] and o["start"] < span["end"] for o in others)


# A word directly followed by a list particle: 사람하고 / 컵만 / 병과 / 개랑 / xyz만.
_LIST_ITEM = re.compile(r"([가-힣A-Za-z]+?)(이랑|하고|과|와|랑|만)(?=[\s,.!?]|$)")
# Two English words joined as list items: "people and cups", "cup, bottle", "cats or dogs".
_ASCII_LIST = re.compile(r"(?<![A-Za-z])([A-Za-z]+)(?=(?:\s*,\s*|\s+(?:and|or)\s+)([A-Za-z]+))")


def _blank(text, spans):
    chars = list(text)
    for span in spans:
        for i in range(span["start"], span["end"]):
            chars[i] = " "
    return "".join(chars)


def _distinct(spans):
    out = []
    for span in spans:
        if span["canonical"] not in out:
            out.append(span["canonical"])
    return out


class CommandNormalizer:
    """Deterministic transcript -> canonical command."""

    def __init__(self, config):
        self.config = config
        self._object = config.matchers["object"]
        self._camera = config.matchers["camera"]
        self._enabled = config.matchers["enabled"]
        self._color = config.matchers["color"]
        self._intent = config.intent_matchers

    # -- evidence --------------------------------------------------------

    def evidence(self, text):
        # "Don't" and "Don’t" are matched as "dont"; the aliases are plain ASCII.
        text = (text or "").strip().replace("’", "").replace("'", "")
        ev = Evidence(text)
        if not text:
            return ev

        camera_spans = self._camera.scan(text)
        ev.cameras = _distinct(camera_spans)
        if "both" in ev.cameras:
            ev.camera = "both"
        elif len(ev.cameras) > 1:
            ev.camera_reason = "ambiguous: %s" % ", ".join(ev.cameras)
        elif ev.cameras:
            ev.camera = ev.cameras[0]
        else:
            ev.camera_reason = "no camera named"

        masked = _blank(text, camera_spans)
        intent_spans = []
        for name in INTENTS:
            source = masked if name in CAMERA_MASKED_INTENTS else text
            spans = self._intent[name].scan(source)
            intent_spans.extend(spans)
            ev.intents[name] = bool(spans)

        # "오렌지" is a class and "오렌지색" a colour: when the sentence is about
        # boxes/colour the colour reading wins, otherwise the object reading does.
        object_spans = self._object.scan(text)
        color_spans = self._color.scan(text)
        if ev.intents["color_target"]:
            object_spans = [s for s in object_spans if not _overlaps(s, color_spans)]
        else:
            color_spans = [s for s in color_spans if not _overlaps(s, object_spans)]
        ev.classes = _distinct(object_spans)
        ev.colors = _distinct(color_spans)
        ev.color = ev.colors[0] if len(ev.colors) == 1 else None

        # Class-list guard. In an "only these objects" context (a class is named, or
        # 만 / only is used), every word joined into the list must be a known phrase.
        # "사람 펀만" names person and something unknown: the request is NOT "person
        # only", and no class is guessed for 펀. A known alias (컷 -> cup) resolves; a
        # filler word (좀, 하나, the, all ...) is not a class candidate.
        if ev.classes or ev.intents["show_only"]:
            known = object_spans + camera_spans + color_spans + intent_spans
            ignore = getattr(self.config, "list_item_ignore", set())
            stopwords = getattr(self.config, "list_item_stopwords", set())

            def unresolved(word, start, end):
                if word in ignore or word.lower() in stopwords:
                    return False
                return not _overlaps({"start": start, "end": end}, known)

            candidates = [(m.group(1), m.start(1), m.end(1)) for m in _LIST_ITEM.finditer(text)]
            for m in _ASCII_LIST.finditer(text):
                pair = [(m.group(1), m.start(1), m.end(1)), (m.group(2), m.start(2), m.end(2))]
                for mine, other in ((pair[0], pair[1]), (pair[1], pair[0])):
                    # an English item counts only when it is joined to a named class
                    if _overlaps({"start": other[1], "end": other[2]}, object_spans):
                        candidates.append(mine)
            for word, start, end in candidates:
                if unresolved(word, start, end) and word not in ev.unresolved_class_terms:
                    ev.unresolved_class_terms.append(word)

        # An ON/OFF word inside a class name ("stop" in "stop sign") is the class.
        enabled_values = _distinct([s for s in self._enabled.scan(text)
                                    if not _overlaps(s, object_spans)])
        if len(enabled_values) == 1:
            ev.enabled = enabled_values[0]
        elif enabled_values:
            ev.enabled_reason = "ambiguous: %s" % ", ".join(enabled_values)
        return ev

    # -- the parser ------------------------------------------------------

    def parse(self, text):
        ev = self.evidence(text)
        if not ev.text:
            return self._unknown(ev, "empty text", "empty")
        i = ev.intents
        classes, color, enabled = ev.classes, ev.color, ev.enabled

        # "사람이 보이면 ... 바꾸고 ..." is a rule, not one command. It is never
        # partly executed.
        if i["conditional"]:
            return self._unknown(ev, "a conditional command (not supported)", "conditional")
        if len(ev.colors) > 1:
            return self._unknown(ev, "more than one colour: %s" % ", ".join(ev.colors),
                                 "colour")

        # swap names two positions and nothing to act on. `바꿔` is also the
        # ordinary verb "change" ("박스를 빨간색으로 바꿔"), so a colour, a class,
        # a reset or an ON/OFF word means it is not a swap.
        if (i["swap"] and color is None and not classes and not i["reset"]
                and not i["show_all"] and enabled is None):
            return self._ok(ev, {"action": "swap_camera"}, "swap phrase")

        if i["reset"]:
            if classes:
                return self._unknown(ev, "a reset phrase together with classes %s"
                                     % classes, "reset")
            return self._with_camera(ev, {"action": "reset"}, "reset phrase")

        if color is not None and not classes:
            if enabled is not None:
                return self._unknown(ev, "a colour together with an ON/OFF word",
                                     "colour")
            return self._with_camera(ev, {"action": "set_box_color", "color": color},
                                     "colour phrase")

        # Camera / display ON-OFF is not a feature of this demo. It is never
        # turned into a detection toggle here.
        if (i["display_target"] and not i["detection_target"] and not classes
                and enabled is not None):
            return Parsed(UNSUPPORTED, self._unknown_command(ev),
                          "camera/display ON-OFF is not supported", "display on/off", ev)

        if enabled is not None and not classes and (i["detection_target"]
                                                    or i["detect_verb"]):
            action = "detection_on" if enabled == "true" else "detection_off"
            return self._with_camera(ev, {"action": action}, "detection ON/OFF phrase")

        if i["show_all"] and not classes and enabled is None:
            return self._with_camera(ev, {"action": "detect_all"}, "show-all phrase")

        if classes and (ev.colors or i["swap"] or i["show_all"]):
            return self._unknown(ev, "classes together with another action", "classes")
        if classes and ev.unresolved_class_terms:
            return self._unknown(ev, "partial class list: %s is not a supported class "
                                 "(unresolved; nothing is guessed)"
                                 % ", ".join(ev.unresolved_class_terms), "classes")
        if classes and enabled is None and (i["show_only"] or i["detect_verb"]
                                            or ev.cameras):
            unknown = [c for c in classes if c not in self.config.canonical_objects]
            if unknown:
                return self._unknown(ev, "not a model class: %s" % unknown, "classes")
            return self._with_camera(ev, {"action": "detect_only",
                                          "classes": list(classes)},
                                     "classes + show/detect phrase")

        if classes and enabled is not None:
            return self._unknown(ev, "classes together with an ON/OFF word", "classes")
        if enabled is None and (i["detection_target"] or i["display_target"]):
            return self._unknown(ev, "no action word the parser knows", "partial")
        return self._unknown(ev, "no supported command recognised", "none")

    # -- results ---------------------------------------------------------

    def _with_camera(self, ev, command, rule):
        if ev.camera is None:
            return self._unknown(ev, "incomplete: %s (a camera is never guessed)"
                                 % ev.camera_reason, rule)
        command = dict(command)
        command["camera"] = ev.camera
        return self._ok(ev, command, rule)

    def _ok(self, ev, command, rule):
        problem = check_command(command, self.config)
        if problem:
            return self._unknown(ev, problem, rule)
        return Parsed(OK, canonical_order(command), None, rule, ev)

    @staticmethod
    def _unknown_command(ev):
        return {"action": "unknown", "original_text": ev.text}

    def _unknown(self, ev, reason, rule):
        return Parsed(UNKNOWN, self._unknown_command(ev), reason, rule, ev)


def canonical_order(command):
    order = ("action", "camera", "classes", "color", "original_text")
    return {k: command[k] for k in order if k in command}


def check_command(command, config):
    """None when `command` is a complete, valid canonical command; else why not.

    The one schema check shared by the parser, the Qwen3 validator, the UI server
    and the manual controls.
    """
    if not isinstance(command, dict):
        return "command is not an object"
    action = command.get("action")
    if action in UNSUPPORTED_ACTIONS:
        return "unsupported action %r" % action
    if action not in SUPPORTED_ACTIONS:
        return "unknown action %r" % (action,)
    fields = SUPPORTED_ACTIONS[action]
    extra = set(command) - {"action"} - set(fields)
    if extra:
        return "%s carries unexpected field(s) %s" % (action, sorted(extra))
    missing = [f for f in fields if f not in command]
    if missing:
        return "%s is missing %s" % (action, missing)
    if "camera" in fields and command["camera"] not in command_config.CAMERAS:
        return "camera must be one of %s" % (list(command_config.CAMERAS),)
    if "classes" in fields:
        classes = command["classes"]
        if not isinstance(classes, list) or not classes:
            return "classes must be a non-empty list"
        bad = [c for c in classes if c not in config.canonical_objects]
        if bad:
            return "not a model class: %s" % bad
        if len(set(classes)) != len(classes):
            return "classes contains duplicates"
    if "color" in fields and command["color"] not in config.colors:
        return "colour must be one of %s" % config.colors
    return None


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

_DEFAULT = {}


def get(config=None):
    if config is not None:
        return CommandNormalizer(config)
    if "n" not in _DEFAULT:
        _DEFAULT["n"] = CommandNormalizer(command_config.load())
    return _DEFAULT["n"]


def parse(text, config=None):
    return get(config).parse(text)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def D(camera, *classes):
    return {"action": "detect_only", "camera": camera, "classes": list(classes)}


def A(action, camera=None, color=None):
    out = {"action": action}
    if camera:
        out["camera"] = camera
    if color:
        out["color"] = color
    return out


CASES = [
    # -- the required seminar commands -------------------------------------
    ("왼쪽 사람만 보여줘", OK, D("left", "person")),
    ("오른쪽 탐지 꺼", OK, A("detection_off", "right")),
    ("왼쪽 카메라 초기화", OK, A("reset", "left")),
    ("왼쪽 카메라 초기화하세요", OK, A("reset", "left")),
    ("왼쪽 카메라 리셋하세요", OK, A("reset", "left")),
    ("왼쪽 박스를 초록색으로 바꿔줘", OK, A("set_box_color", "left", "green")),
    ("왼쪽 카메라에서 사람하고 컵만 탐지해 주세요", OK, D("left", "person", "cup")),
    ("왼쪽 카메라 아무것도 탐지하지 마세요", OK, A("detection_off", "left")),
    ("왼쪽 바운딩 박스를 그린으로 바꿔주세요", OK, A("set_box_color", "left", "green")),
    ("좌우 카메라 바꿔", OK, A("swap_camera")),
    # -- multi-class ------------------------------------------------------
    ("왼쪽 카메라에서 사람과 컵만 탐지해 주세요", OK, D("left", "person", "cup")),
    ("왼쪽 사람하고 컵 보여줘", OK, D("left", "person", "cup")),
    ("왼쪽에서 person, cup, bottle만 보여줘", OK, D("left", "person", "cup", "bottle")),
    ("오른쪽에서 컵이랑 사람만 보여줘", OK, D("right", "cup", "person")),
    ("Show only people and cups on the left", OK, D("left", "person", "cup")),
    ("오른쪽에서 휴대폰만 찾아", OK, D("right", "cell phone")),
    ("왼쪽에서 말만 보여줘", OK, D("left", "horse")),
    ("오른쪽 오렌지만 보여줘", OK, D("right", "orange")),
    ("양쪽 사람만 보여줘", OK, D("both", "person")),
    # -- detection OFF / ON ------------------------------------------------
    ("왼쪽 탐지하지 마", OK, A("detection_off", "left")),
    ("왼쪽 탐지 꺼", OK, A("detection_off", "left")),
    ("오른쪽 detection off", OK, A("detection_off", "right")),
    ("오른쪽 탐지를 꺼주세요", OK, A("detection_off", "right")),
    ("Turn detection off on the right", OK, A("detection_off", "right")),
    ("Dont detect anything on the left", OK, A("detection_off", "left")),
    ("Don't detect anything on the left", OK, A("detection_off", "left")),
    ("왼쪽 탐지 켜", OK, A("detection_on", "left")),
    ("Turn detection on on the right", OK, A("detection_on", "right")),
    ("양쪽 탐지 꺼", OK, A("detection_off", "both")),
    # -- colour -----------------------------------------------------------
    ("오른쪽 박스를 빨간색으로 해줘", OK, A("set_box_color", "right", "red")),
    ("make the left bounding boxes green", OK, A("set_box_color", "left", "green")),
    ("왼쪽 카메라 박스 색상을 노란색으로", OK, A("set_box_color", "left", "yellow")),
    ("오른쪽 박스를 오렌지색으로 바꿔", OK, A("set_box_color", "right", "orange")),
    ("박스 색을 파란색으로", UNKNOWN, None),
    ("왼쪽 박스를 빨간색 파란색으로", UNKNOWN, None),
    # -- reset and show-all -------------------------------------------------
    ("Reset the left camera", OK, A("reset", "left")),
    ("오른쪽 카메라 초기화 하세요", OK, A("reset", "right")),
    ("오른쪽 원래대로", OK, A("reset", "right")),
    ("왼쪽 전체 클래스 보여줘", OK, A("detect_all", "left")),
    ("오른쪽 필터 해제", OK, A("detect_all", "right")),
    ("Show all classes on the right", OK, A("detect_all", "right")),
    ("초기화", UNKNOWN, None),
    # -- swap -------------------------------------------------------------
    ("좌우 카메라를 바꿔주세요", OK, A("swap_camera")),
    ("왼쪽 오른쪽 카메라 교체하세요", OK, A("swap_camera")),
    ("카메라 교체하세요", OK, A("swap_camera")),
    ("Swap the cameras", OK, A("swap_camera")),
    ("Switch left and right", OK, A("swap_camera")),
    # -- not decided by the parser (these go to Qwen3) -----------------------
    ("오른쪽 카메라 꺼", UNSUPPORTED, None),
    ("Turn the right camera off", UNSUPPORTED, None),
    ("자동차만 보여주세요", UNKNOWN, None),
    ("개를 탐지해 주세요", UNKNOWN, None),
    ("왼쪽 박스 없애줘", UNKNOWN, None),
    ("왼쪽 오른쪽 사람만 보여줘", UNKNOWN, None),
    ("왼쪽 사람 탐지 꺼", UNKNOWN, None),
    ("오른쪽에서 늑대만 보여주세요", UNKNOWN, None),
    ("사람이 보이면 오른쪽 카메라로 바꾸고 컵이 나오면 빨간 박스를 그려줘", UNKNOWN, None),
    ("If you see a person switch to the right camera", UNKNOWN, None),
    ("왼쪽 사람만 빨간색으로 보여줘", UNKNOWN, None),
    # -- measured Whisper spellings, and the partial-list guard ----------------
    ("왼쪽 카메라에서 사람하고 컷만 탐지해주세요.", OK, D("left", "person", "cup")),
    ("오른쪽 팜지 꺼", OK, A("detection_off", "right")),
    ("왼쪽 카메라에서 사람하고 곰돌이만 보여줘", UNKNOWN, None),
    # -- class-list guard: unresolved class terms (regressions) -----------------
    ("왼쪽 카메라에서 사람하고 컵만 탐지해 주세요", OK, D("left", "person", "cup")),
    ("왼쪽 카메라 사람 펀만 탐지해 주세요", UNKNOWN, None),
    ("왼쪽 카메라 사람 컷만 탐지해 주세요", OK, D("left", "person", "cup")),
    ("왼쪽 펀만 보여줘", UNKNOWN, None),
    ("왼쪽에서 person, cup, xyz만 보여줘", UNKNOWN, None),
    ("Show only people and blorps on the left", UNKNOWN, None),
    # filler words are not class candidates
    ("왼쪽 카메라에서 사람만 좀 보여주세요", OK, D("left", "person")),
    ("왼쪽 사람 하나만 보여줘", OK, D("left", "person")),
    ("Show only people and the cups on the left", OK, D("left", "person", "cup")),
    ("이번만 오른쪽 탐지 꺼", OK, A("detection_off", "right")),
    ("왼쪽에서 사람과 머그잔만 보여줘", UNKNOWN, None),
    ("왼쪽에서 사과만 보여줘", OK, D("left", "apple")),
    ("왼쪽만 사람 보여줘", OK, D("left", "person")),
    ("왼쪽 사람만 보여주고 카메라 바꿔", UNKNOWN, None),
    ("나도 모르겠어요", UNKNOWN, None),
    ("오늘 날씨 어때", UNKNOWN, None),
    ("", UNKNOWN, None),
]

CLASS_CASES = [
    (["개", "개를", "개만", "강아지", "강아지를", "dog", "dogs"], "dog"),
    (["사람", "사람을", "사람만", "사람들", "person", "people"], "person"),
    (["휴대폰", "핸드폰만", "스마트폰을", "cell phone", "phones"], "cell phone"),
    (["컵", "컵만", "컵과", "컵하고", "컵이랑", "cup", "cups"], "cup"),
]


def _selftest():
    config = command_config.load()
    normalizer = CommandNormalizer(config)
    failures = []

    def check(label, cond, extra=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                               "" if cond else "\n         -> " + str(extra)))
        if not cond:
            failures.append(label)

    print("command_normalizer.py self-test\n")
    for text, expect_status, expect_command in CASES:
        result = normalizer.parse(text)
        good = result.status == expect_status
        if good and expect_command is not None:
            good = result.command == expect_command
        if good and expect_command is None:
            good = result.command.get("action") == "unknown"
        check("%-44r -> %s" % (text, json.dumps(expect_command, ensure_ascii=False)
                               if expect_command else expect_status), good,
              "%s %s (%s)" % (result.status,
                              json.dumps(result.command, ensure_ascii=False),
                              result.reason))

    print("\n-- unresolved class terms are preserved in the parse result")
    for text, expect in (("왼쪽 카메라 사람 펀만 탐지해 주세요", ["펀"]),
                         ("Show only people and blorps on the left", ["blorps"]),
                         ("왼쪽 카메라 사람 컷만 탐지해 주세요", []),
                         ("왼쪽 카메라에서 사람만 좀 보여주세요", []),
                         ("왼쪽 박스 없애줘", [])):
        parsed = normalizer.parse(text)
        check("%r -> unresolved %s" % (text, expect),
              parsed.evidence.unresolved_class_terms == expect
              and parsed.as_dict()["evidence"]["unresolved_class_terms"] == expect,
              parsed.evidence.unresolved_class_terms)

    print("\n-- class aliases")
    for phrases, expected in CLASS_CASES:
        bad = []
        for phrase in phrases:
            got = normalizer.parse("오른쪽에서 %s 보여줘" % phrase)
            if got.command.get("classes") != [expected]:
                bad.append("%s->%s" % (phrase, got.command))
        check("%d phrase(s) -> %s" % (len(phrases), expected), not bad, bad)

    unreachable = [n for n in config.canonical_objects
                   if normalizer.parse("오른쪽에서 %s 보여줘" % n).command.get("classes") != [n]]
    check("%d/%d classes reachable by their English name"
          % (len(config.canonical_objects) - len(unreachable),
             len(config.canonical_objects)), not unreachable, unreachable)

    print("\n-- the schema check")
    check("a valid multi-class command passes",
          check_command(D("left", "person", "cup"), config) is None)
    for bad in ({"action": "display", "camera": "right"},
                {"action": "detect_only", "camera": "left"},
                {"action": "detect_only", "camera": "left", "classes": []},
                {"action": "detect_only", "camera": "left", "classes": ["wolf"]},
                {"action": "detect_only", "camera": "up", "classes": ["person"]},
                {"action": "set_box_color", "camera": "left", "color": "ultraviolet"},
                {"action": "reset", "camera": "left", "color": "red"},
                "reset"):
        check("refused: %s" % json.dumps(bad, ensure_ascii=False),
              check_command(bad, config) is not None)

    if failures:
        print("\n%d check(s) FAILED" % len(failures))
        return 1
    print("\nall checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("text", nargs="*", help="a transcript to parse")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if not args.text:
        parser.error("give a transcript, or --selftest")
    try:
        result = parse(" ".join(args.text))
    except command_config.ConfigError as exc:
        sys.exit("ERROR: %s" % exc)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
