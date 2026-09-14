#!/usr/bin/env python3
"""
panel_state.py - the state of each camera view and the view <-> screen-position mapping.

    python3 backend/panel_state.py --selftest

WHAT IT HOLDS

    source_mapping      which SOURCE (camera view, channel) is shown in the LEFT / RIGHT panel
    detection_enabled   per SOURCE - does that view draw overlays
    classes             per SOURCE - None (every class) or a list of class names the
                        view draws, e.g. ["person", "cup"]
    box_color           per SOURCE - one colour for every box, or None for the
                        per-class palette

ONE CAMERA VIEW IS ONE UNIT

A source is one camera view: its video, its metadata (the same channel), its
detection / segmentation overlay, its model label, its capture status and the
overlay state above. A swap changes source_mapping only, so every part of a view
moves to the other side together; nothing of a view is left behind at a position.

LEFT / RIGHT / both in a command mean the view currently shown there: the position is
resolved to its source when the command is applied, and the change stays with that
source across later swaps.

Every action is drawing-only. Nothing here can reach a camera, a stream, a model or
the MLA: the strongest thing a command changes is what web/overlay.js paints.

RESET

`reset` restores the COMPLETE startup state of the view shown at that position -
detection ON, every class, default box colour. It does not undo a swap, which is the
layout of both panels rather than a property of one camera view.

THE LOCK

Held for a dict read or write and nothing else: no HTTP call, no parsing, no voice
request and no SSE write happens while it is held.
"""

import argparse
import copy
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import command_normalizer                                        # noqa: E402

LEFT = "left"
RIGHT = "right"
BOTH = "both"
POSITIONS = (LEFT, RIGHT)

# The manual controls' way back to the per-class palette. No sentence produces it.
AUTO_COLOR = "auto"

# The stroke for each canonical colour name of command_config.json's `color`
# slot. web/overlay.js has the same table; scripts/selftest.py compares them.
BOX_COLORS = {
    "red": "#ef4444",
    "green": "#22c55e",
    "blue": "#3b82f6",
    "yellow": "#facc15",
    "white": "#f8fafc",
    "orange": "#fb923c",
    "purple": "#a855f7",
    "cyan": "#22d3ee",
    "pink": "#f472b6",
    "black": "#0b0f14",
}


def default_panel():
    """The startup state of one camera view."""
    return {"detection_enabled": True, "classes": None, "box_color": None}


class CommandResult:
    """What one command did."""

    def __init__(self, status, message, changed=False):
        self.status = status
        self.message = message
        self.changed = changed

    @property
    def ok(self):
        return self.status == "applied"

    def as_dict(self):
        return {"status": self.status, "message": self.message,
                "changed": self.changed}

    def __repr__(self):
        return "CommandResult(%s, %r)" % (self.status, self.message)


class PanelState:
    """Per-source (camera view) render state plus the position -> source mapping."""

    def __init__(self, config, panels=None):
        self._config = config
        self._lock = threading.RLock()
        defaults = dict(panels or config.display_defaults)
        self._source = {LEFT: str(defaults[LEFT]), RIGHT: str(defaults[RIGHT])}
        self._initial = dict(self._source)
        # Keyed by SOURCE, never by position: a view's overlay state belongs to the
        # view and moves with it when the panels are swapped.
        self._state = {self._source[position]: default_panel() for position in POSITIONS}
        self._swapped = False
        # `version` moves only when something a panel renders changed, so a
        # no-op command makes no client re-render.
        self.version = 1
        self.counters = {"applied_commands": 0, "rejected_commands": 0,
                         "ignored_commands": 0, "class_changes": 0,
                         "detection_toggles": 0, "swaps": 0, "resets": 0,
                         "color_changes": 0}
        self.last_change_at = time.time()

    # ----------------------------------------------------------------- reads

    def source_of_position(self, position):
        with self._lock:
            return self._source[position]

    def render_plan(self):
        """What each SOURCE must render, keyed by source (one WebRTC connection each)."""
        with self._lock:
            plan = {}
            for position in POSITIONS:
                state = self._state[self._source[position]]
                plan[self._source[position]] = {
                    "position": position,
                    "detection_enabled": state["detection_enabled"],
                    "classes": list(state["classes"]) if state["classes"] else None,
                    "box_color": state["box_color"],
                }
            return plan

    def snapshot(self):
        with self._lock:
            out = {"version": self.version,
                   "swapped": self._swapped,
                   "initial_mapping": dict(self._initial),
                   "mapping": dict(self._source),
                   "counters": dict(self.counters),
                   "last_change_at": self.last_change_at,
                   "positions": {}, "sources": {}}
            for position in POSITIONS:
                channel = self._source[position]
                state = copy.deepcopy(self._state[channel])
                task = self._config.stream_task(channel)
                state["source_stream"] = channel
                state["source_task"] = task
                state["metadata_type"] = ("segmentation" if task == "segmentation"
                                          else "object-detection")
                state["supported_class_count"] = len(self._config.stream_classes(channel))
                state["filter_supported"] = (
                    state["classes"] is None
                    or all(self._config.stream_supports(channel, c)
                           for c in state["classes"]))
                state["is_default"] = (
                    {k: state[k] for k in default_panel()} == default_panel())
                out["positions"][position] = state
                view = dict(state)
                view.pop("source_stream")
                view["position"] = position
                out["sources"][channel] = view
            return out

    # ---------------------------------------------------------------- writes

    def apply_command(self, command):
        """Apply ONE canonical command (backend/command_normalizer.py format)."""
        if not isinstance(command, dict):
            return self._reject("command is not a JSON object")
        action = command.get("action")

        if action == "unknown":
            with self._lock:
                self.counters["ignored_commands"] += 1
            return CommandResult("ignored", "Command not understood")

        manual_auto = action == "set_box_color" and command.get("color") == AUTO_COLOR
        checked = dict(command, color="red") if manual_auto else command
        problem = command_normalizer.check_command(checked, self._config)
        if problem:
            return self._reject(problem)

        if action == "swap_camera":
            with self._lock:
                self._source[LEFT], self._source[RIGHT] = (
                    self._source[RIGHT], self._source[LEFT])
                self._swapped = not self._swapped
                self.counters["applied_commands"] += 1
                self.counters["swaps"] += 1
                self._touch()
                left, right = self._source[LEFT], self._source[RIGHT]
            return CommandResult("applied", "Swapped: LEFT <- source %s, RIGHT <- source %s"
                                 % (left, right), changed=True)

        camera = command["camera"]
        positions = list(POSITIONS) if camera == BOTH else [camera]
        label = "BOTH" if camera == BOTH else camera.upper()

        if action == "detect_only":
            classes = list(command["classes"])
            with self._lock:
                for position in positions:
                    channel = self._source[position]
                    missing = [c for c in classes
                               if not self._config.stream_supports(channel, c)]
                    if missing:
                        self.counters["rejected_commands"] += 1
                        return CommandResult(
                            "unsupported_for_current_source",
                            "Source %s does not emit %s. Nothing changed."
                            % (channel, ", ".join(missing)))
                new = {"detection_enabled": True, "classes": classes}
                changed = self._update(positions, new)
                self.counters["applied_commands"] += 1
                if changed:
                    self.counters["class_changes"] += 1
            return CommandResult("applied", "%s: detecting %s" % (label, " + ".join(classes)),
                                 changed=changed)

        if action == "detect_all":
            with self._lock:
                changed = self._update(positions, {"detection_enabled": True,
                                                   "classes": None})
                self.counters["applied_commands"] += 1
                if changed:
                    self.counters["class_changes"] += 1
            return CommandResult("applied", "%s: detecting all classes" % label,
                                 changed=changed)

        if action in ("detection_on", "detection_off"):
            enabled = action == "detection_on"
            with self._lock:
                changed = self._update(positions, {"detection_enabled": enabled})
                self.counters["applied_commands"] += 1
                if changed:
                    self.counters["detection_toggles"] += 1
            return CommandResult("applied", "%s: detection %s"
                                 % (label, "ON" if enabled else "OFF"), changed=changed)

        if action == "reset":
            with self._lock:
                changed = self._update(positions, default_panel())
                self.counters["applied_commands"] += 1
                self.counters["resets"] += 1
            return CommandResult("applied", "%s: reset to the startup state "
                                 "(detection ON, all classes, default box colour)" % label,
                                 changed=changed)

        if action == "set_box_color":
            color = None if manual_auto else command["color"]
            if color is not None and color not in BOX_COLORS:
                return self._reject("no stroke for colour %r" % color)
            with self._lock:
                changed = self._update(positions, {"box_color": color})
                self.counters["applied_commands"] += 1
                if changed:
                    self.counters["color_changes"] += 1
            return CommandResult("applied", "%s: box colour %s"
                                 % (label, color or "default"), changed=changed)

        return self._reject("unsupported action %r" % action)

    # -- internals --------------------------------------------------------

    def _update(self, positions, values):
        """Caller holds the lock. True when anything a panel renders changed.

        Each position is resolved to the source (camera view) shown there right now,
        and the change is stored with that source, so it moves with the view on a swap.
        """
        changed = False
        for position in positions:
            view = self._state[self._source[position]]
            for key, value in values.items():
                value = list(value) if isinstance(value, list) else value
                if view[key] != value:
                    view[key] = value
                    changed = True
        if changed:
            self._touch()
        return changed

    def _reject(self, message):
        with self._lock:
            self.counters["rejected_commands"] += 1
        return CommandResult("rejected", message)

    def _touch(self):
        self.version += 1
        self.last_change_at = time.time()


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _selftest():
    import command_config

    config = command_config.load()
    failures = []

    def check(label, cond, extra=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                               "" if cond else "   <- " + str(extra)))
        if not cond:
            failures.append(label)

    def pos(state, position):
        return state.snapshot()["positions"][position]

    def core(state, position):
        p = pos(state, position)
        return {k: p[k] for k in default_panel()}

    print("panel_state.py self-test")
    state = PanelState(config)
    check("left starts on source 0, right on source 1",
          state.source_of_position(LEFT) == "0" and state.source_of_position(RIGHT) == "1")
    check("both positions start in the default state",
          core(state, LEFT) == default_panel() and core(state, RIGHT) == default_panel())

    # -- multi-class --------------------------------------------------------
    r = state.apply_command({"action": "detect_only", "camera": LEFT,
                             "classes": ["person", "cup"]})
    check("detect_only with two classes applies", r.ok and r.changed, r)
    check("the class list is stored as a list",
          state.render_plan()["0"]["classes"] == ["person", "cup"], state.render_plan())
    r = state.apply_command({"action": "detect_only", "camera": LEFT,
                             "classes": ["person", "cup", "bottle"]})
    check("three classes replace two",
          pos(state, LEFT)["classes"] == ["person", "cup", "bottle"], pos(state, LEFT))
    check("RIGHT untouched", core(state, RIGHT) == default_panel())

    # -- the spec's reset sequence, for LEFT and for RIGHT ---------------------
    for side in (LEFT, RIGHT):
        state.apply_command({"action": "set_box_color", "camera": side, "color": "green"})
        state.apply_command({"action": "detection_off", "camera": side})
        state.apply_command({"action": "detect_only", "camera": side, "classes": ["person"]})
        before = core(state, side)
        check("%s set up: green, restricted to person" % side.upper(),
              before["box_color"] == "green" and before["classes"] == ["person"], before)
        state.apply_command({"action": "detection_off", "camera": side})
        check("%s detection OFF keeps classes and colour" % side.upper(),
              core(state, side) == {"detection_enabled": False, "classes": ["person"],
                                    "box_color": "green"}, core(state, side))
        r = state.apply_command({"action": "reset", "camera": side})
        check("%s reset restores detection ON, all classes, default colour" % side.upper(),
              r.ok and core(state, side) == default_panel() and pos(state, side)["is_default"],
              core(state, side))

    # -- both ----------------------------------------------------------------
    state.apply_command({"action": "detection_off", "camera": BOTH})
    check("detection_off both",
          not pos(state, LEFT)["detection_enabled"] and not pos(state, RIGHT)["detection_enabled"])
    state.apply_command({"action": "reset", "camera": BOTH})
    check("reset both", core(state, LEFT) == default_panel() and core(state, RIGHT) == default_panel())

    # -- detect_only turns detection back on; detect_all clears the list -----
    state.apply_command({"action": "detection_off", "camera": RIGHT})
    state.apply_command({"action": "detect_only", "camera": RIGHT, "classes": ["dog"]})
    check("detect_only makes the chosen classes visible",
          pos(state, RIGHT)["detection_enabled"] is True)
    state.apply_command({"action": "set_box_color", "camera": RIGHT, "color": "red"})
    state.apply_command({"action": "detect_all", "camera": RIGHT})
    check("detect_all clears classes, keeps the colour",
          core(state, RIGHT) == {"detection_enabled": True, "classes": None, "box_color": "red"},
          core(state, RIGHT))

    # -- unknown / rejected are true no-ops -------------------------------------
    for bad in ({"action": "unknown", "original_text": "x"},
                {"action": "display", "camera": RIGHT},
                {"action": "detect_only", "camera": LEFT, "classes": ["wolf"]},
                {"action": "detect_only", "camera": LEFT, "object": "person"},
                {"action": "set_box_color", "camera": LEFT, "color": "ultraviolet"},
                {"action": "reset"}):
        before = state.snapshot()
        r = state.apply_command(bad)
        after = state.snapshot()
        check("no-op: %s -> %s" % (bad, r.status),
              r.status in ("ignored", "rejected") and not r.changed
              and after["positions"] == before["positions"]
              and after["version"] == before["version"], r)

    # -- a swap moves each camera view as ONE unit --------------------------------
    # Source 0 (LEFT) gets a cup filter; source 1 (RIGHT) has the red colour from above.
    state.apply_command({"action": "detect_only", "camera": LEFT, "classes": ["cup"]})
    plan_before = state.render_plan()
    snap_before = state.snapshot()
    state.apply_command({"action": "swap_camera"})
    snap = state.snapshot()
    plan = state.render_plan()
    check("swap: source 0 is now on the RIGHT with its cup filter",
          snap["positions"][RIGHT]["source_stream"] == "0"
          and snap["positions"][RIGHT]["classes"] == ["cup"]
          and plan["0"] == dict(plan_before["0"], position=RIGHT), (snap["positions"], plan))
    check("swap: source 1 is now on the LEFT with its red colour",
          snap["positions"][LEFT]["source_stream"] == "1"
          and snap["positions"][LEFT]["box_color"] == "red"
          and plan["1"] == dict(plan_before["1"], position=LEFT), (snap["positions"], plan))
    check("swap: nothing of a view stays at its old position",
          core(state, LEFT) == {k: snap_before["positions"][RIGHT][k] for k in default_panel()}
          and core(state, RIGHT) == {k: snap_before["positions"][LEFT][k] for k in default_panel()})
    check("snapshot.sources reports each view's state and current side",
          snap["sources"]["0"]["position"] == RIGHT and snap["sources"]["0"]["classes"] == ["cup"]
          and snap["sources"]["1"]["position"] == LEFT and snap["sources"]["1"]["box_color"] == "red",
          snap["sources"])

    # A command addresses the view shown at that position NOW, and stays with it.
    state.apply_command({"action": "detection_off", "camera": LEFT})
    check("after the swap, LEFT means source 1",
          state.render_plan()["1"]["detection_enabled"] is False
          and state.render_plan()["0"]["detection_enabled"] is True)
    state.apply_command({"action": "swap_camera"})
    check("swap back: source 1 is RIGHT again and still has detection OFF and red",
          core(state, RIGHT) == {"detection_enabled": False, "classes": None, "box_color": "red"}
          and core(state, LEFT) == {"detection_enabled": True, "classes": ["cup"], "box_color": None},
          (core(state, LEFT), core(state, RIGHT)))
    state.apply_command({"action": "detection_on", "camera": RIGHT})
    check("swap back: the original mapping and every view's state are restored",
          state.snapshot()["mapping"] == snap_before["mapping"]
          and state.render_plan() == plan_before, (state.render_plan(), plan_before))

    state.apply_command({"action": "swap_camera"})
    r = state.apply_command({"action": "reset", "camera": LEFT})
    check("reset after a swap resets the view on the LEFT (source 1) and keeps the swap",
          r.ok and state.snapshot()["mapping"] == {LEFT: "1", RIGHT: "0"}
          and core(state, LEFT) == default_panel()
          and core(state, RIGHT) == {"detection_enabled": True, "classes": ["cup"], "box_color": None},
          (core(state, LEFT), core(state, RIGHT)))

    state.apply_command({"action": "set_box_color", "camera": RIGHT, "color": "blue"})
    r = state.apply_command({"action": "set_box_color", "camera": RIGHT, "color": AUTO_COLOR})
    check("manual 'auto' returns to the per-class palette",
          r.ok and pos(state, RIGHT)["box_color"] is None, r)
    check("every configured colour has a stroke",
          set(config.colors) <= set(BOX_COLORS), sorted(set(config.colors) - set(BOX_COLORS)))

    if failures:
        print("\n%d check(s) FAILED" % len(failures))
        return 1
    print("\nall checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    parser.error("nothing to do; try --selftest")


if __name__ == "__main__":
    sys.exit(main())
