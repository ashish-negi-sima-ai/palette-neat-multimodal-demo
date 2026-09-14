#!/usr/bin/env python3
"""
command_config.py - loader, validator and phrase matcher for command_config.json.

command_config.json is the single source of truth for this demo (spec section
21): the real YOLO class sets discovered from the model archives, the Korean and
English aliases, the supported actions and the intent phrases.  It is never
replaced by a built-in COCO list - a missing or invalid file is a hard, explicit
startup error (section 61).

    python3 command_config.py --selftest
    python3 command_config.py --summary

WHO USES IT
-----------
backend/command_normalizer.py (the deterministic parser), backend/command_processor.py
(the Qwen3 fallback validator), backend/panel_state.py (class and colour checks) and
voice/board_voice_server.py (the Qwen3 few-shot examples) all load this one file.

THE THREE MATCHING RULES (spec sections 17, 18, 24, 25)
-------------------------------------------------------
1. ASCII aliases match on an ASCII word boundary, spelled out rather than \\b so
   that a Korean particle may follow an English noun: "왼쪽에서 bus만 보여줘".
2. Hangul aliases of two or more syllables match as plain substrings.  A Korean
   particle is simply whatever follows the alias, so 개를 / 강아지만 / 사람들을
   need no stripping and none is done (section 18).
3. Hangul aliases of ONE syllable are the dangerous case - 개 is also the Korean
   counter word - so they match only when bounded by non-Hangul on both sides,
   never after a number or counter, and (when listed in particle_required) only
   when a recognized particle follows.  Nothing is ever stripped blindly.

Longest phrase wins over non-overlapping spans, so "cell phone" beats "phone",
"hot dog" beats "dog", and "turn off" beats "off".  There is no fuzzy matching,
no edit distance and no nearest-class guessing anywhere (section 25).
"""

import argparse
import hashlib
import json
import os
import re
import sys

DEFAULT_PATH = os.environ.get(
    "COMMAND_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "command_config.json"))

# Spec section 2: the actions a state/routing handler in this demo may receive.
# A configuration may not invent actions and may not re-add a removed one.
#
# Every action is drawing-only: it changes what the browser paints over the
# picture (web/overlay.js) and touches no stream, camera or model.
#
#   detect_only    draw only these classes (a list, any length)
#   detect_all     draw every class again (detection ON, class filter cleared)
#   detection_on   draw overlays for this camera
#   detection_off  draw no overlays for this camera
#   reset          the camera's complete startup state: detection ON, all
#                  classes, default box colour
#   set_box_color  one box colour for every class of this camera
#   swap_camera    exchange the LEFT and RIGHT panels
#
# `camera` is left, right or both.
SUPPORTED_ACTIONS = {
    "swap_camera":   [],
    "detect_only":   ["camera", "classes"],
    "detect_all":    ["camera"],
    "detection_on":  ["camera"],
    "detection_off": ["camera"],
    "reset":         ["camera"],
    "set_box_color": ["camera", "color"],
    "unknown":       ["original_text"],
}

CAMERAS = ("left", "right", "both")

# Spec sections 2-4: recognised so the gate can name them in a log line and in
# the GUI, never executable.  They are deliberately NOT in SUPPORTED_ACTIONS,
# have no handler anywhere in this project, and may not appear in the config.
UNSUPPORTED_ACTIONS = {
    "display": "camera/display ON/OFF is unsupported by design",
}

SLOTS = ("camera", "object", "color", "enabled")

# `enabled` is a closed two-value polarity slot whose Korean forms are imperative
# verb endings (켜 / 꺼 / 끄).  They carry the whole meaning of the command and
# are kept as whisper validated them, so the noun-slot syllable rules
# do not apply to them.
VERB_SLOTS = ("enabled",)

_ASCII_ONLY = re.compile(r"^[A-Za-z0-9 \-]+$")
_HANGUL = re.compile(r"[가-힣]")


class ConfigError(Exception):
    """command_config.json is missing, malformed, empty or inconsistent."""


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _lookbehind_guard(words):
    """Negative lookbehinds that keep a one-syllable noun out of a count.

    Python's `re` needs every lookbehind to be fixed width, so this emits one
    per guard word rather than a single alternation.  It is what stops "몇 개",
    "두 개" and "3개" from being read as the class `dog` (spec section 18).
    """
    out = [r"(?<![가-힣])", r"(?<![0-9])"]
    for word in words:
        out.append(r"(?<!%s )" % re.escape(word))
    return "".join(out)


def compile_phrase(phrase, policy):
    """The regex for one alias, following the three rules in the module docstring.

    Returns (regex, kind) where kind is ascii / hangul / hangul_short - the GUI
    and the self-test report it so the matching rule in force is never a guess.
    """
    if _ASCII_ONLY.match(phrase):
        # English has one genuinely ambiguous pair: the polarity words `on` and
        # `off` are also the preposition in "on the right".  Without this guard
        # "Turn detection off on the right" resolves to both true and false at
        # once and the whole command is demoted to unknown.  The exclusions are
        # data, in matching.ascii.not_followed_by, not a special case in code.
        blocked = (policy.get("not_followed_by") or {}).get(phrase.lower(), [])
        tail = "".join(r"(?!%s)" % re.escape(s) for s in blocked)
        return (re.compile(r"(?<![A-Za-z0-9_])" + re.escape(phrase)
                           + r"(?![A-Za-z0-9_])" + tail, re.IGNORECASE), "ascii")

    syllables = len(_HANGUL.findall(phrase))
    if syllables >= int(policy.get("short_alias_max_syllables", 1)) + 1:
        return re.compile(re.escape(phrase)), "hangul"

    # One-syllable Hangul noun: bounded, number-guarded, particle-aware.
    particles = sorted(policy.get("particles", []), key=len, reverse=True)
    alternation = "|".join(re.escape(p) for p in particles) or r"(?!)"
    required = phrase in set(policy.get("particle_required", []))
    tail = "(?:%s)%s" % (alternation, "" if required else "?")
    return (re.compile(_lookbehind_guard(policy.get("number_guard", []))
                       + re.escape(phrase)
                       + "(?P<particle>%s)" % tail
                       + r"(?![가-힣])"), "hangul_short")


class Matcher:
    """Deterministic phrase matching for one slot (spec sections 17, 24, 25).

    scan() reduces a transcript to NON-OVERLAPPING alias spans, longest phrase
    first at each position.  Each span also reports the Korean particle that
    followed the alias, so particle handling is observable rather than implied.
    """

    def __init__(self, table, policy):
        self._entries = []                       # (phrase, canonical, rx, kind)
        for canonical, phrases in table.items():
            for phrase in phrases:
                rx, kind = compile_phrase(phrase, policy)
                self._entries.append((phrase, canonical, rx, kind))
        self._entries.sort(key=lambda e: len(e[0]), reverse=True)
        self.canonicals = sorted(table.keys())
        self._policy = policy

    def scan(self, text):
        found = []
        for phrase, canonical, rx, kind in self._entries:
            for m in rx.finditer(text):
                particle = ""
                if kind == "hangul_short":
                    particle = m.groupdict().get("particle") or ""
                elif kind == "hangul":
                    particle = _trailing_particle(text, m.end(), self._policy)
                found.append((m.start(), -(m.end() - m.start()), m.end(),
                              canonical, phrase, kind, particle))
        found.sort()
        spans = []
        cursor = 0
        for start, _neg, end, canonical, phrase, kind, particle in found:
            if start < cursor:
                continue                         # overlapped by a longer phrase
            spans.append({"start": start, "end": end, "canonical": canonical,
                          "phrase": phrase, "kind": kind, "particle": particle})
            cursor = end
        return spans

    def resolve(self, text):
        """(value, reason).  value is None whenever the speech is not decisive.

        Two different canonicals in one utterance is ambiguity, and ambiguity is
        `unknown` - never a pick between them (spec section 25).
        """
        ordered = []
        for span in self.scan(text):
            if span["canonical"] not in ordered:
                ordered.append(span["canonical"])
        if not ordered:
            return None, "nothing supported was spoken"
        if len(ordered) > 1:
            return None, "ambiguous: %s" % ", ".join(ordered)
        return ordered[0], None

    def canonicalize(self, value):
        """Map a value some model volunteered onto a canonical one, or None.

        Used only to record how close an advisory parameter was; it decides
        nothing (spec section 10).
        """
        if not isinstance(value, str):
            return None
        needle = value.strip().lower()
        for phrase, canonical, _rx, _kind in self._entries:
            if phrase.lower() == needle:
                return canonical
        return None


def _trailing_particle(text, end, policy):
    """The recognized Korean particle directly after a matched alias, or "".

    Reported, never removed: the alias already matched without it (spec section
    18).  It exists so a self-test can assert that 개를 really was read as the
    alias 개 followed by the particle 를.
    """
    rest = text[end:]
    for particle in sorted(policy.get("particles", []), key=len, reverse=True):
        if rest.startswith(particle):
            after = rest[len(particle):]
            if not after or not _HANGUL.match(after[0]):
                return particle
    return ""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _require(cond, message):
    if not cond:
        raise ConfigError(message)


def _validate_alias_list(slot, canonical, phrases, policy):
    _require(isinstance(phrases, list) and phrases,
             "slots.%s.%s must be a non-empty list of aliases" % (slot, canonical))
    short_max = int(policy.get("short_alias_max_syllables", 1))
    for phrase in phrases:
        _require(isinstance(phrase, str) and phrase.strip(),
                 "slots.%s.%s contains an empty or non-string alias"
                 % (slot, canonical))
        if _HANGUL.search(phrase):
            syllables = len(_HANGUL.findall(phrase))
            _require(slot in VERB_SLOTS or syllables >= 1,
                     "slots.%s.%s: %r has no Hangul syllable" % (slot, canonical, phrase))
            if slot not in VERB_SLOTS and syllables <= short_max:
                # A guarded short alias is allowed, but only if the guard can
                # actually be built for it.
                _require(policy.get("particles"),
                         "slots.%s.%s: short alias %r needs matching.particles"
                         % (slot, canonical, phrase))
        else:
            _require(_ASCII_ONLY.match(phrase),
                     "slots.%s.%s: alias %r is neither plain ASCII nor Hangul"
                     % (slot, canonical, phrase))
    _require(canonical in phrases or slot == "enabled",
             "slots.%s.%s must list its own canonical name as an alias"
             % (slot, canonical))


class CommandConfig:
    """A validated command_config.json plus the matchers built from it."""

    def __init__(self, data, path, raw_text):
        self.path = path
        self.data = data
        self.sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        self._validate()

        self.streams = data["streams"]
        self.canonical_objects = data["canonical_objects"]
        self.slots = data["slots"]
        self.actions = data["actions"]
        self.intent = data["intent"]
        self.few_shot = data["few_shot"]
        self.display_defaults = data["display_defaults"]
        self.matching = data["matching"]
        # Class-list guard vocabulary: filler words that are never class candidates.
        list_items = data["matching"].get("list_items") or {}
        self.list_item_ignore = set(list_items.get("ignore") or [])
        self.list_item_stopwords = {w.lower() for w in (list_items.get("ascii_stopwords") or [])}
        self.provenance = data.get("provenance", {})
        # One policy object for both scripts, so a Matcher needs no slot context.
        self.policy = dict(data["matching"]["hangul"])
        self.policy["not_followed_by"] = (
            (data["matching"].get("ascii") or {}).get("not_followed_by", {}))

        self.matchers = {slot: Matcher(self.slots[slot], self.policy)
                         for slot in SLOTS}
        self.intent_matchers = {name: Matcher(table, self.policy)
                                for name, table in self.intent.items()}

    # -- validation ------------------------------------------------------

    def _validate(self):
        data = self.data
        _require(isinstance(data, dict), "top level must be a JSON object")
        for field in ("config_version", "streams", "canonical_objects", "slots",
                      "actions", "intent", "few_shot", "display_defaults",
                      "matching"):
            _require(field in data, "missing required field %r" % field)
        _require(data["config_version"] == 2,
                 "unsupported config_version %r (this build understands 2)"
                 % data["config_version"])
        _require(isinstance(data["matching"].get("hangul"), dict),
                 "matching.hangul must be an object")
        policy = data["matching"]["hangul"]

        # -- streams: the real, discovered model class sets ---------------
        streams = data["streams"]
        _require(isinstance(streams, dict) and streams,
                 "streams must be a non-empty object")
        for channel, entry in streams.items():
            _require(isinstance(entry, dict), "streams.%s must be an object" % channel)
            for field in ("task", "num_classes", "classes", "discovered_from"):
                _require(field in entry, "streams.%s is missing %r" % (channel, field))
            classes = entry["classes"]
            _require(isinstance(classes, list) and classes,
                     "streams.%s.classes must be a non-empty list" % channel)
            _require(len(classes) == int(entry["num_classes"]),
                     "streams.%s: num_classes is %s but %d classes are listed"
                     % (channel, entry["num_classes"], len(classes)))
            _require(len(set(classes)) == len(classes),
                     "streams.%s.classes contains duplicates" % channel)

        union = data["canonical_objects"]
        _require(isinstance(union, list) and union,
                 "canonical_objects must be a non-empty list")
        expected = set()
        for entry in streams.values():
            expected.update(entry["classes"])
        _require(set(union) == expected,
                 "canonical_objects is not the union of the stream class sets "
                 "(only in union: %s; only in streams: %s)"
                 % (sorted(set(union) - expected), sorted(expected - set(union))))

        # -- slots ---------------------------------------------------------
        slots = data["slots"]
        _require(isinstance(slots, dict), "slots must be an object")
        for slot in SLOTS:
            _require(slot in slots, "slots.%s is missing" % slot)
            _require(isinstance(slots[slot], dict) and slots[slot],
                     "slots.%s must be a non-empty object" % slot)
            for canonical, phrases in slots[slot].items():
                _validate_alias_list(slot, canonical, phrases, policy)

        _require(set(slots["camera"]) == set(CAMERAS),
                 "slots.camera must define exactly %s, found %s"
                 % (list(CAMERAS), sorted(slots["camera"])))
        _require(set(slots["enabled"]) == {"true", "false"},
                 "slots.enabled must define exactly true and false, found %s"
                 % sorted(slots["enabled"]))
        _require(set(slots["object"]) == set(union),
                 "slots.object must cover exactly the canonical object set "
                 "(missing: %s; extra: %s)"
                 % (sorted(set(union) - set(slots["object"])),
                    sorted(set(slots["object"]) - set(union))))

        # Spec section 25: one alias may never resolve to two canonicals.
        for slot in SLOTS:
            owner = {}
            for canonical, phrases in slots[slot].items():
                for phrase in phrases:
                    key = phrase.lower()
                    _require(owner.get(key, canonical) == canonical,
                             "slots.%s: alias %r maps to both %r and %r"
                             % (slot, phrase, owner.get(key), canonical))
                    owner[key] = canonical

        # -- actions: the gate's allowlist, in the file ---------------------
        actions = data["actions"]
        _require(isinstance(actions, dict), "actions must be an object")
        for name in UNSUPPORTED_ACTIONS:
            _require(name not in actions,
                     "actions must not contain %r: %s (spec sections 2-3)"
                     % (name, UNSUPPORTED_ACTIONS[name]))
        _require(set(actions) == set(SUPPORTED_ACTIONS),
                 "actions must be exactly %s, found %s"
                 % (sorted(SUPPORTED_ACTIONS), sorted(actions)))
        for action, fields in SUPPORTED_ACTIONS.items():
            entry = actions[action]
            _require(isinstance(entry, dict) and "fields" in entry,
                     "actions.%s must be an object with a fields list" % action)
            _require(list(entry["fields"]) == fields,
                     "actions.%s.fields must be %s, found %s"
                     % (action, fields, entry["fields"]))

        # -- intent phrase tables ------------------------------------------
        intent = data["intent"]
        _require(isinstance(intent, dict), "intent must be an object")
        for name in ("show_only", "detect_verb", "detection_target",
                     "display_target", "swap", "reset", "show_all", "color_target",
                     "conditional"):
            _require(name in intent and isinstance(intent[name], dict)
                     and intent[name], "intent.%s must be a non-empty object" % name)
            for canonical, phrases in intent[name].items():
                _require(isinstance(phrases, list) and phrases,
                         "intent.%s.%s must be a non-empty list" % (name, canonical))

        # -- few-shot (the Qwen3 examples) ----------------------------------
        # Each answer is a complete structured command in the exact shape the
        # validator accepts, so the model is never shown an answer it would be
        # refused for.
        few_shot = data["few_shot"]
        _require(isinstance(few_shot, list) and few_shot,
                 "few_shot must be a non-empty list")
        for pair in few_shot:
            _require(isinstance(pair, list) and len(pair) == 2,
                     "few_shot entries must be [text, object] pairs")
            text, obj = pair
            _require(isinstance(text, str) and text.strip(),
                     "few_shot text must be a non-empty string")
            _require(isinstance(obj, dict) and obj.get("action") in SUPPORTED_ACTIONS,
                     "few_shot answer has no supported action: %r" % (obj,))
            wanted = set(SUPPORTED_ACTIONS[obj["action"]]) - {"original_text"}
            _require(set(obj) - {"action"} == wanted,
                     "few_shot answer for %r must carry exactly %s, found %r"
                     % (text, sorted(wanted), obj))
            if "camera" in obj:
                _require(obj["camera"] in CAMERAS, "few_shot camera %r" % obj["camera"])
            if "classes" in obj:
                _require(isinstance(obj["classes"], list) and obj["classes"]
                         and set(obj["classes"]) <= set(union),
                         "few_shot classes %r" % (obj["classes"],))
            if "color" in obj:
                _require(obj["color"] in slots["color"], "few_shot color %r" % obj["color"])

        # -- display defaults ------------------------------------------------
        defaults = data["display_defaults"]
        _require(isinstance(defaults, dict) and set(defaults) == {"left", "right"},
                 "display_defaults must define exactly left and right")
        for position, channel in defaults.items():
            _require(channel in streams,
                     "display_defaults.%s points at unknown stream %r"
                     % (position, channel))
        _require(defaults["left"] != defaults["right"],
                 "display_defaults maps both positions to the same stream")

    # -- lookups -----------------------------------------------------------

    def stream_classes(self, channel):
        entry = self.streams.get(str(channel))
        return list(entry["classes"]) if entry else []

    def stream_supports(self, channel, object_name):
        """Spec section 51: does THIS model really emit that class?"""
        return object_name in set(self.stream_classes(channel))

    def stream_task(self, channel):
        entry = self.streams.get(str(channel))
        return entry["task"] if entry else "unknown"

    @property
    def colors(self):
        return sorted(self.slots["color"].keys())

    def resolve(self, slot, text):
        return self.matchers[slot].resolve(text)

    def summary(self):
        lines = ["command_config.json : %s" % self.path,
                 "sha256              : %s" % self.sha256]
        for channel in sorted(self.streams):
            entry = self.streams[channel]
            lines.append("stream %s (%-12s): %d classes  [%s]"
                         % (channel, entry["task"], entry["num_classes"],
                            entry["discovered_from"]))
        sets = [tuple(sorted(e["classes"])) for e in self.streams.values()]
        lines.append("same class sets     : %s"
                     % ("yes" if len(set(sets)) == 1 else "NO"))
        lines.append("canonical objects   : %d" % len(self.canonical_objects))
        lines.append("object alias phrases: %d"
                     % sum(len(v) for v in self.slots["object"].values()))
        lines.append("supported actions   : %s" % " | ".join(sorted(self.actions)))
        lines.append("unsupported actions : %s"
                     % " | ".join(sorted(UNSUPPORTED_ACTIONS)))
        return "\n".join(lines)


def load(path=None):
    """Read, parse and validate command_config.json.  Never falls back.

    Spec sections 21 and 61: a missing, empty, malformed or inconsistent
    configuration is a hard, explicit failure.  There is no built-in COCO list
    anywhere in this project to fall back to.
    """
    path = path or DEFAULT_PATH
    if not os.path.isfile(path):
        raise ConfigError(
            "command_config.json not found at %s\n"
            "       It is the shared single source of truth and must exist\n"
            "       before anything starts (it ships in backend/)." % path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ConfigError("cannot read %s: %s" % (path, exc))
    if not raw.strip():
        raise ConfigError("%s is empty" % path)
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ConfigError("%s is not valid JSON: %s" % (path, exc))
    return CommandConfig(data, path, raw)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _selftest(cfg):
    failures = []

    def check(label, cond, extra=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                               "" if cond else "   <- " + str(extra)))
        if not cond:
            failures.append(label)

    print("command_config.py self-test")
    check("config_version is 2", cfg.data["config_version"] == 2)
    check("actions are exactly the supported ones",
          set(cfg.actions) == set(SUPPORTED_ACTIONS), sorted(cfg.actions))
    for name in UNSUPPORTED_ACTIONS:
        check("%r is absent from the config" % name, name not in cfg.actions)

    objects = cfg.matchers["object"]
    for text, expected in (("개를 탐지해 주세요", "dog"),
                           ("강아지만 보여주세요", "dog"),
                           ("사람들을 보여줘", "person"),
                           ("휴대폰만 보여줘", "cell phone"),
                           ("show only dogs on the left", "dog"),
                           ("hot dogs please", "hot dog")):
        value, reason = objects.resolve(text)
        check("%r -> %s" % (text, expected), value == expected, value or reason)

    for text in ("사과 세 개를 주세요", "3개만 보여줘"):
        value, _ = objects.resolve(text)
        check("counter guard: %r is not read as dog" % text, value != "dog", value)

    spans = objects.scan("개를 보여줘")
    check("개를 is alias 개 + particle 를",
          spans and spans[0]["phrase"] == "개" and spans[0]["particle"] == "를",
          spans)

    if failures:
        print("\n%d check(s) FAILED" % len(failures))
        return 1
    print("\nall checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=DEFAULT_PATH)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    try:
        cfg = load(args.config)
    except ConfigError as exc:
        sys.exit("ERROR: %s" % exc)
    if args.selftest:
        return _selftest(cfg)
    print(cfg.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
