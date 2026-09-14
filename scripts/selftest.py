#!/usr/bin/env python3
"""
selftest.py - offline proof of the behaviours this demo is specified around.

    ./run.sh --selftest
    python3 scripts/selftest.py
    python3 scripts/selftest.py --live [https://127.0.0.1:8022]   # also the running server

No microphone and no DevKit needed (without --live). What it checks:

  A  architecture     this app opens no UDP socket; the page opens each source once
  B  parser           the required seminar commands through the deterministic parser
  C  command path     voice and typed text call the SAME command_processor.process();
                      Qwen3 answers are strictly validated (scripted answers)
  D  panel state      multi-class lists, complete reset, detection OFF, both cameras,
                      a command that is not understood changes nothing
  E  overlay rule     class-list filtering changes list membership only
  F  renderer         Insight's drawing.js is the one used; colour tables agree
  G  the page         title, MLA workload list, no "Decided by", text input, font size
  H  browser half     node scripts/render_test.js and media_isolation_test.js
"""

import argparse
import json
import os
import re
import ssl
import subprocess
import shutil
import sys
from urllib import request as urlrequest

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
BACKEND = os.path.join(APP_DIR, "backend")
WEB = os.path.join(APP_DIR, "web")
sys.path.insert(0, BACKEND)

import command_config                                            # noqa: E402
import command_normalizer                                        # noqa: E402
import command_processor                                         # noqa: E402
import panel_state                                               # noqa: E402

FAILURES = []


def check(label, condition, extra=""):
    print("  [%s] %s%s" % ("PASS" if condition else "FAIL", label,
                           "" if condition else "\n         -> %s" % (extra,)))
    if not condition:
        FAILURES.append(label)


def section(title):
    print("\n%s %s" % (title, "-" * max(0, 66 - len(title))))


def read(*parts):
    with open(os.path.join(APP_DIR, *parts), "r", encoding="utf-8") as handle:
        return handle.read()


def code_only(text):
    """Python source with comments and docstrings removed (for pattern checks)."""
    import ast
    import io
    import tokenize
    lines = text.splitlines()
    drop = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                drop.update(range(first.lineno, first.end_lineno + 1))
    cut = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            cut[token.start[0]] = token.start[1]
    return "\n".join(line[:cut.get(n, len(line))] for n, line in enumerate(lines, 1)
                     if n not in drop)


# --------------------------------------------------------------------------

def test_architecture():
    section("A  no media transport in this application")
    ours = "\n".join(code_only(read("backend", name)) for name in sorted(os.listdir(BACKEND))
                     if name.endswith(".py"))
    check("no datagram socket is created anywhere", "SOCK_DGRAM" not in ours)
    check("no socket bind anywhere", not re.search(r"\.bind\(", ours))
    app_js = read("web", "app.js")
    check("the page opens each source exactly once", app_js.count("InsightSource.open") == 1)
    check("no command handler in the page touches a connection",
          not re.search(r"(sendCommand|applyState|renderRecord)[\s\S]{0,900}?"
                        r"(conn\.stop|InsightSource\.open|srcObject\s*=)", app_js))
    check("the swap is a CSS grid column and nothing else",
          'setProperty("--col"' in app_js)


REQUIRED = [
    ("왼쪽 사람만 보여줘", {"action": "detect_only", "camera": "left", "classes": ["person"]}),
    ("오른쪽 탐지 꺼", {"action": "detection_off", "camera": "right"}),
    ("왼쪽 카메라 초기화", {"action": "reset", "camera": "left"}),
    ("왼쪽 박스를 초록색으로 바꿔줘", {"action": "set_box_color", "camera": "left", "color": "green"}),
    ("왼쪽 카메라에서 사람하고 컵만 탐지해 주세요",
     {"action": "detect_only", "camera": "left", "classes": ["person", "cup"]}),
    ("왼쪽 카메라 아무것도 탐지하지 마세요", {"action": "detection_off", "camera": "left"}),
    ("왼쪽 바운딩 박스를 그린으로 바꿔주세요",
     {"action": "set_box_color", "camera": "left", "color": "green"}),
    ("좌우 카메라 바꿔", {"action": "swap_camera"}),
    ("왼쪽 카메라 초기화하세요", {"action": "reset", "camera": "left"}),
    ("Reset the right camera", {"action": "reset", "camera": "right"}),
    ("왼쪽에서 person, cup, bottle만 보여줘",
     {"action": "detect_only", "camera": "left", "classes": ["person", "cup", "bottle"]}),
    ("오른쪽 detection off", {"action": "detection_off", "camera": "right"}),
    ("make the left bounding boxes green",
     {"action": "set_box_color", "camera": "left", "color": "green"}),
]


def test_parser(config):
    section("B  the deterministic parser on the required commands")
    normalizer = command_normalizer.CommandNormalizer(config)
    for text, expected in REQUIRED:
        parsed = normalizer.parse(text)
        check("%-40s -> %s" % (text, json.dumps(expected, ensure_ascii=False)),
              parsed.ok and parsed.command == expected,
              "%s %s (%s)" % (parsed.status, parsed.command, parsed.reason))
    off = normalizer.parse("오른쪽 카메라 꺼")
    check("camera OFF is not a detection toggle", not off.ok and off.status == "unsupported")
    return normalizer


def test_command_path(normalizer):
    section("C  one command path for voice and text; Qwen3 answers validated")
    voice = code_only(read("voice", "board_voice_server.py"))
    server = code_only(read("backend", "server.py"))
    check("the voice runtime runs recognised speech through command_processor.process",
          "command_processor.process(" in voice and "self.process_text(stt_text)" in voice)
    check("the voice runtime runs typed text through the same process_text",
          "self.process_text(text)" in voice and "/text_command" in voice)
    check("the UI server has no parser of its own for typed text (fallback = same function)",
          "command_processor.process(text, self.normalizer, None)" in server
          and ".parse(" not in server)
    check("Qwen3 runs inside the vision MLA lease", 'self.gate.hold("voice_qwen")' in voice)

    def scripted(answer):
        return lambda text: (answer, None)
    r = command_processor.process("왼쪽 박스 없애줘", normalizer,
                                  scripted('{"action":"detection_off","camera":"left"}'))
    check("a valid Qwen3 answer is executed and marked (Qwen3)",
          r["status"] == "ok" and r["display"] == "LEFT / DETECTION OFF (Qwen3)", r["display"])
    r = command_processor.process("왼쪽 사람만 보여줘", normalizer,
                                  scripted('{"action":"detection_off","camera":"left"}'))
    check("a parser command never shows (Qwen3)",
          r["source"] == "parser" and "(Qwen3)" not in r["display"])
    # class-list guard regressions
    r = command_processor.process("왼쪽 카메라에서 사람하고 컵만 탐지해 주세요", normalizer,
                                  scripted('{"action":"detect_only","camera":"left","classes":["person"]}'))
    check("class list 1: person + cup -> LEFT / PERSON + CUP", r["display"] == "LEFT / PERSON + CUP")
    r = command_processor.process("왼쪽 카메라 사람 펀만 탐지해 주세요", normalizer,
                                  scripted('{"action":"detect_only","camera":"left","classes":["person"]}'))
    check("class list 2: person + unknown 펀 -> Command not understood (Qwen3 PERSON ONLY refused)",
          r["status"] == "not_understood" and r["display"] == "Command not understood", r["display"])
    r = command_processor.process("왼쪽 카메라 사람 컷만 탐지해 주세요", normalizer, None)
    check("class list 3: person + alias 컷 -> LEFT / PERSON + CUP", r["display"] == "LEFT / PERSON + CUP")
    r = command_processor.process("왼쪽 사람이랑 컵만 켜줘", normalizer,
                                  scripted('{"action":"detect_only","camera":"left","classes":["person"]}'))
    check("class list 4: a Qwen3 subset of the requested classes is rejected",
          r["status"] == "not_understood", r["display"])
    for answer in ('```json {"action":"detection_off","camera":"left"}```',
                   '{"action":"detection_off","camera":"right"}',
                   '{"action":"detection_off"}',
                   '{"action":"display","camera":"left"}'):
        r = command_processor.process("왼쪽 박스 없애줘", normalizer, scripted(answer))
        check("refused: %s" % answer, r["status"] == "not_understood"
              and r["display"] == "Command not understood")


def test_state(config):
    section("D  panel state")
    state = panel_state.PanelState(config)
    state.apply_command({"action": "set_box_color", "camera": "left", "color": "green"})
    state.apply_command({"action": "detection_off", "camera": "left"})
    state.apply_command({"action": "detect_only", "camera": "left", "classes": ["person"]})
    state.apply_command({"action": "detection_off", "camera": "left"})
    state.apply_command({"action": "reset", "camera": "left"})
    left = state.snapshot()["positions"]["left"]
    check("reset: detection ON, all classes, default colour",
          left["detection_enabled"] is True and left["classes"] is None
          and left["box_color"] is None, left)
    state.apply_command({"action": "detect_only", "camera": "right",
                         "classes": ["person", "cup", "bottle"]})
    check("the class list is stored as a list",
          state.render_plan()["1"]["classes"] == ["person", "cup", "bottle"])
    # A swap moves each camera view as one unit: its overlay state goes with it.
    swap = panel_state.PanelState(config)
    swap.apply_command({"action": "detect_only", "camera": "left", "classes": ["person"]})
    swap.apply_command({"action": "set_box_color", "camera": "left", "color": "green"})
    swap.apply_command({"action": "set_box_color", "camera": "right", "color": "red"})
    plan_before = swap.render_plan()
    swap.apply_command({"action": "swap_camera"})
    plan_after = swap.render_plan()
    check("swap: each source keeps its own overlay state and changes side",
          all({k: v for k, v in plan_after[s].items() if k != "position"}
              == {k: v for k, v in plan_before[s].items() if k != "position"}
              and plan_after[s]["position"] != plan_before[s]["position"] for s in ("0", "1")),
          (plan_before, plan_after))
    swap.apply_command({"action": "swap_camera"})
    check("swap twice restores the mapping and every view's state",
          swap.render_plan() == plan_before, swap.render_plan())

    before = state.snapshot()
    result = state.apply_command({"action": "unknown", "original_text": "오늘 날씨 어때"})
    after = state.snapshot()
    check("not understood changes nothing",
          result.status == "ignored" and after["positions"] == before["positions"]
          and after["version"] == before["version"])


def overlay_filter(message, plan):
    key = {"object-detection": "objects", "segmentation": "segments"}[message["type"]]
    entries = message["data"][key]
    if not plan["detection_enabled"]:
        return None, len(entries), 0
    if not plan["classes"]:
        return message, len(entries), len(entries)
    kept = [e for e in entries if e["label"] in plan["classes"]]
    return dict(message, data=dict(message["data"], **{key: kept})), len(entries), len(kept)


def test_overlay_rule():
    section("E  class-list filtering")
    message = {"type": "object-detection", "data": {"objects": [
        {"label": "person"}, {"label": "cup"}, {"label": "dog"}, {"label": "bottle"}]}}
    original = json.dumps(message)
    out, total, kept = overlay_filter(message, {"detection_enabled": True,
                                                "classes": ["person", "cup"]})
    check("person + cup keeps 2 of 4", (kept, total) == (2, 4)
          and [e["label"] for e in out["data"]["objects"]] == ["person", "cup"])
    check("the arriving message is not mutated", json.dumps(message) == original)
    overlay = read("web", "overlay.js")
    check("overlay.js filters by a class list", "plan.classes" in overlay)


def test_renderer(config_path):
    section("F  the overlay renderer is the installed Insight's")
    import server
    vf_base = server.load_ui_config(config_path)["insight"]["vf_base"]
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with urlrequest.urlopen(vf_base.rstrip("/") + "/static/drawing.js", timeout=8,
                                context=context) as response:
            body = response.read().decode("utf-8", "replace")
        check("vf serves drawing.js with window.drawStrategies", "window.drawStrategies" in body)
    except Exception as exc:                                    # noqa: BLE001
        print("  SKIP vf not reachable from here (%s)" % exc)
    check("this project keeps no copy of drawing.js", not os.path.exists(os.path.join(WEB, "drawing.js")))
    overlay = read("web", "overlay.js")
    block = re.search(r"var BOX_COLORS = \{(.*?)\};", overlay, re.S)
    js_colors = dict(re.findall(r'(\w+)\s*:\s*"(#[0-9a-fA-F]{6})"', block.group(1))) if block else {}
    check("JS and Python colour tables agree", js_colors == panel_state.BOX_COLORS,
          (js_colors, panel_state.BOX_COLORS))
    check("every configured colour is drawable",
          set(command_config.load().colors) <= set(js_colors))


def test_page():
    section("G  the page")
    page = read("web", "index.html")
    check("title is Palette NEAT SDK Demo",
          "<title>Palette NEAT SDK Demo</title>" in page and "<h1>Palette NEAT SDK Demo</h1>" in page)
    check("no DRONE AI DEMO", "DRONE AI DEMO" not in page.upper().replace("PALETTE", ""))
    check("the MLA statement", "4 AI workloads running concurrently on" in page and "Modalix MLA" in page)
    names = re.findall(r'<span class="wl-name">([^<]+)</span>', page)
    check("the four MLA workloads, and only those",
          names == ["YOLO26m Detection", "YOLO26m Segmentation", "Whisper-medium", "Qwen3-0.6B"], names)
    check("the parser is not listed as a workload", "parser" not in " ".join(names).lower())
    check("no Decided by field", "decided by" not in page.lower()
          and "decided_by" not in read("web", "app.js"))
    check("a text command input with Run", 'id="textInput"' in page and 'id="textRun"' in page)
    check("language selector Auto | KO | EN",
          all('data-lang="%s"' % lang in page for lang in ("auto", "ko", "en")))
    check("font size 100 / 125 / 150", all('data-scale="%s"' % s in page for s in ("1", "1.25", "1.5")))
    check("capture mode is not hard-coded in the page",
          "720p" not in page and "720p" not in read("web", "app.js"))
    app_js = read("web", "app.js")
    check("one Clear button for Recognized / Command",
          page.count('id="clearBtn"') == 1 and page.count('class="btn clear"') == 1)
    check("KO / Auto placeholders",
          all(t in app_js for t in ("음성 입력을 기다립니다.", "명령을 기다립니다.", "듣는 중...")))
    check("EN placeholders",
          all(t in app_js for t in ("Waiting for voice input...", "Waiting for command...", "Listening...")))
    check("the page starts in the configured language (no saved choice is read)",
          'store("demo.language")' not in app_js
          and json.loads(read("config", "ui_config.json"))["voice"]["language"] == "ko")
    server = read("backend", "server.py")
    clear = server.split("def clear_display", 1)[1].split("\n    def ", 1)[0]
    check("Clear only empties the display record (no panel state, voice or vision call)",
          "self.last_command.set()" in clear
          and not any(w in clear for w in ("self.state", "self.voice", "apply_")), clear)


def test_browser_side():
    section("H  the browser half (node)")
    if shutil.which("node") is None:
        print("  SKIP node is not installed")
        return
    for script, minimum in (("render_test.js", 30), ("media_isolation_test.js", 15)):
        result = subprocess.run(["node", os.path.join(HERE, script)], capture_output=True, text=True)
        passed, failed = result.stdout.count("[PASS]"), result.stdout.count("[FAIL]")
        check("%s: %d checks" % (script, passed),
              result.returncode == 0 and failed == 0 and passed >= minimum,
              result.stderr.strip()[-300:] or "%d failed" % failed)


def test_live(base):
    section("I  the running server")
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(base + path, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=60, context=context) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        check("the server answers", call("/api/health").get("status") == "ok")
    except Exception as exc:                                    # noqa: BLE001
        check("the server answers at %s" % base, False, exc)
        return
    start = call("/api/state")["state"]
    reply = call("/api/text/command", {"text": "왼쪽 카메라에서 사람하고 컵만 탐지해 주세요"})
    check("typed multi-class command applies",
          reply["state"]["positions"]["left"]["classes"] == ["person", "cup"]
          and reply["command"]["display"] == "LEFT / PERSON + CUP", reply["command"])
    before = reply["state"]
    reply = call("/api/text/command", {"text": "오늘 날씨 어때"})
    check("not understood leaves the state unchanged",
          reply["command"]["display"] == "Command not understood"
          and reply["state"]["positions"] == before["positions"], reply["command"])
    reply = call("/api/text/command", {"text": "왼쪽 카메라 초기화하세요"})
    check("reset restores the left default",
          reply["state"]["positions"]["left"]["is_default"], reply["state"]["positions"]["left"])
    call("/api/text/command", {"text": "왼쪽 사람만 보여줘"})
    before = call("/api/state")["state"]
    cleared = call("/api/command/clear", {})
    after = call("/api/state")
    check("Clear empties the displayed command and changes no panel state",
          cleared["command"]["display"] is None and after["command"]["input"] is None
          and after["state"]["positions"] == before["positions"]
          and after["state"]["mapping"] == before["mapping"]
          and after["state"]["version"] == before["version"], after["command"])
    call("/api/text/command", {"text": "왼쪽 카메라 초기화하세요"})
    config = call("/api/config")
    check("config reports languages auto/ko/en", config["voice"]["languages"] == ["ko", "en", "auto"]
          or set(config["voice"]["languages"]) == {"ko", "en", "auto"})
    check("the workloads endpoint answers", "vision" in call("/api/workloads"))
    check("mapping unchanged by these checks", call("/api/state")["state"]["mapping"] == start["mapping"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--live", nargs="?", const="auto")
    args = parser.parse_args()
    print("Palette NEAT SDK Demo self-test")
    config = command_config.load()
    config_path = os.path.join(APP_DIR, "config", "ui_config.json")
    test_architecture()
    normalizer = test_parser(config)
    test_command_path(normalizer)
    test_state(config)
    test_overlay_rule()
    test_renderer(config_path)
    test_page()
    test_browser_side()
    if args.live:
        base = args.live
        if base == "auto":
            ui = json.loads(read("config", "ui_config.json"))
            base = "https://127.0.0.1:%d" % ui["web"]["port"]
        test_live(base.rstrip("/"))
    print("")
    if FAILURES:
        print("%d check(s) FAILED" % len(FAILURES))
        for label in FAILURES:
            print("  - %s" % label)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
