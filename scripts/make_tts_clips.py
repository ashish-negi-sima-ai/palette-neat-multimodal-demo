#!/usr/bin/env python3
"""
make_tts_clips.py - spoken test clips for the command acceptance tests.

    python3 scripts/make_tts_clips.py            # create the clips that are missing
    python3 scripts/make_tts_clips.py --force    # re-create all of them
    python3 scripts/make_tts_clips.py --list

Development tool, run on the SDK host (it needs internet access). Reads
testdata/commands/manifest.json and writes one WAV per entry next to it:

    gtts-cli (Google Translate TTS, Korean or English voice) -> mp3
    ffmpeg                                                   -> 16 kHz mono 16-bit WAV

gtts-cli is looked up on PATH, then in the SDK's neat-insight venv
(/opt/neat-insight/venv/bin/gtts-cli), or set GTTS_CLI.

A TTS clip is a clean, evenly paced reading. It proves the pipeline can hear the
phrase; it does not prove a room full of people can be heard over a fan.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(APP_DIR, "testdata", "commands")


def gtts_cli():
    for candidate in (os.environ.get("GTTS_CLI"), shutil.which("gtts-cli"),
                      "/opt/neat-insight/venv/bin/gtts-cli"):
        if candidate and os.path.isfile(candidate):
            return candidate
    sys.exit("ERROR: gtts-cli not found (set GTTS_CLI)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    entries = json.load(open(os.path.join(CLIPS, "manifest.json"), encoding="utf-8"))
    if args.list:
        for e in entries:
            print("%-34s %s  %s" % (e["file"], e["lang"], e["text"]))
        return 0
    cli = gtts_cli()
    made = 0
    for e in entries:
        wav = os.path.join(CLIPS, e["file"])
        if os.path.isfile(wav) and not args.force:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = os.path.join(tmp, "clip.mp3")
            command = [cli, "-l", e["lang"], "-o", mp3]
            if e.get("slow"):
                command.append("--slow")
            if e.get("tld"):
                command += ["-t", e["tld"]]
            subprocess.run(command + [e["text"]], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ac", "1",
                            "-ar", "16000", "-sample_fmt", "s16",
                            "-af", "adelay=250|250,apad=pad_dur=0.3", wav], check=True)
        made += 1
        print("wrote %s  (%s)" % (os.path.relpath(wav, APP_DIR), e["text"]))
    print("%d clip(s) written, %d in the manifest" % (made, len(entries)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
