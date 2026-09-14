#!/usr/bin/env python3
"""
qwen_probe.py - Qwen3-0.6B (Modalix MLA) as the fallback command parser, on its own.

    ~/pyneat/bin/python3 scripts/qwen_probe.py                     # built-in phrases
    ~/pyneat/bin/python3 scripts/qwen_probe.py --force-qwen        # Qwen3 for every phrase
    ~/pyneat/bin/python3 scripts/qwen_probe.py "왼쪽 박스 없애줘" ...

Development and test tool, run ON THE DEVKIT while the demo is NOT running (it
loads its own Qwen3-0.6B and takes no vision MLA lease). Every phrase goes through
backend/command_processor.process() with the real model, exactly as the voice
runtime does; --force-qwen also asks the model for phrases the parser decides, to
see how its structured answers validate.
"""

import argparse
import json
import os
import sys
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "backend"))
sys.path.insert(0, os.path.join(APP_DIR, "voice"))
sys.path.insert(0, os.path.join(APP_DIR, "scripts"))

import command_processor                                         # noqa: E402
from board_voice_server import CommandSpec, QWEN_MAX_NEW_TOKENS  # noqa: E402
from check_models import model_root                              # noqa: E402

PHRASES = [
    "왼쪽 사람만 보여줘",
    "오른쪽 탐지 꺼",
    "왼쪽 카메라 초기화",
    "왼쪽 박스를 초록색으로 바꿔줘",
    "왼쪽 카메라에서 사람하고 컵만 탐지해 주세요",
    "왼쪽 카메라 아무것도 탐지하지 마세요",
    "왼쪽 바운딩 박스를 그린으로 바꿔주세요",
    "좌우 카메라 바꿔",
    # phrasings the parser does not decide
    "왼쪽 박스 없애줘",
    "오른쪽 박스 좀 숨겨줘",
    "Hide the boxes on the right",
    "Remove the boxes from the left camera",
    "오른쪽 카메라 기본 상태로 돌려줘",
    "오른쪽 카메라 처음 상태로 돌려줘",
    "Make the right camera like it was at the start",
    "왼쪽 사람이랑 컵만 켜줘",
    "왼쪽 화면 박스 지워줘",
    # not supported
    "오늘 날씨 어때",
    "오른쪽 카메라 꺼",
    "사람이 보이면 오른쪽 카메라로 바꾸고 컵이 나오면 빨간 박스를 그려줘",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("phrases", nargs="*")
    ap.add_argument("--force-qwen", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", help="write the results as JSON here")
    args = ap.parse_args()

    import pyneat as neat
    manifest = json.load(open(os.path.join(APP_DIR, "config", "model_manifest.json")))
    qwen_dir = os.path.join(model_root(), manifest["models"]["qwen3_0_6b"]["path"])
    spec = CommandSpec(os.path.join(APP_DIR, "backend", "command_config.json"),
                       os.path.join(APP_DIR, "voice", "command_prompt.txt"))
    t0 = time.time()
    llm = neat.genai.GenAIModel(qwen_dir)
    print("Qwen3-0.6B %s open on the MLA in %d ms" % (llm.model_id(), (time.time() - t0) * 1000))

    def qwen(text):
        req = neat.genai.GenerationRequest()
        req.messages = spec.messages(neat, text)
        req.max_new_tokens = QWEN_MAX_NEW_TOKENS
        req.enable_thinking = False
        t = time.time()
        result = llm.run(req)
        qwen.last_ms = int((time.time() - t) * 1000)
        return result.text, None

    results = []
    for _ in range(args.repeat):
        for text in (args.phrases or PHRASES):
            r = command_processor.process(text, spec.normalizer, qwen)
            forced = None
            if args.force_qwen and r["source"] == "parser":
                raw, _ = qwen(text)
                obj, why = command_processor.strict_json_object(raw)
                cmd = None
                if obj is not None:
                    cmd, why = command_processor.validate_model_command(
                        obj, spec.normalizer.evidence(text), spec.config)
                forced = {"raw": raw, "valid": cmd, "why": why, "ms": qwen.last_ms,
                          "agrees_with_parser": cmd == r["command"]}
            row = {"text": text, "display": r["display"], "source": r["source"],
                   "command": r["command"], "qwen_raw": r["qwen_raw"],
                   "qwen_ms": r["timing_ms"]["qwen"], "reason": r["reason"], "forced": forced}
            results.append(row)
            print("%-52s -> %-34s [%s%s]" % (text[:52], r["display"], r["source"] or "-",
                                            (" qwen %d ms raw=%r" % (r["timing_ms"]["qwen"], r["qwen_raw"][:90]))
                                            if r["qwen_invoked"] else ""))
            if r["reason"]:
                print("      reason: %s" % r["reason"])
            if forced:
                print("      forced Qwen3: %d ms valid=%s agrees=%s raw=%r"
                      % (forced["ms"], json.dumps(forced["valid"], ensure_ascii=False),
                         forced["agrees_with_parser"], forced["raw"][:90]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
