#!/usr/bin/env python3
"""
portability_audit.py - is this project self-contained?

    python3 scripts/portability_audit.py [--workspace /workspace] [--json out.json]

Read-only. Reports:

  symlinks          any symbolic link inside the project (there must be none)
  other projects    every file that mentions another directory of the workspace,
                    split into
                      PATH      an absolute path such as /workspace/<other>/...  (a runtime
                                dependency candidate - must be zero)
                      NAME      the bare directory name, e.g. in a comment (provenance)
  binary            absolute paths compiled into the vision binary
  models            where config/model_manifest.json says each model lives

The allowed external, project-specific dependencies are the two GenAI model
directories under MODEL_ROOT. Everything else must be in this directory or be part
of the installed SiMa SDK / OS.
"""

import argparse
import json
import os
import re
import subprocess
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK_DIRS = {".claude", ".ghcr.io-sima-neat-sdk-v2.1.3.0", ".insight-media", "sima-neat",
            "sima-neat-0.2.2-Linux-extras"}
TEXT_LIMIT = 8 * 1024 * 1024


def is_text(path):
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(4096)
    except OSError:
        return False
    return b"\0" not in chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--workspace", default=os.path.dirname(APP_DIR))
    ap.add_argument("--json")
    args = ap.parse_args()
    me = os.path.basename(APP_DIR)
    others = sorted(d for d in os.listdir(args.workspace)
                    if os.path.isdir(os.path.join(args.workspace, d))
                    and d != me and d not in SDK_DIRS)

    symlinks, path_hits, name_hits, files = [], [], [], 0
    # Bare-name matching only for distinctive names (drone_seminar_USB, whisper_yolo, demo-neat);
    # generic ones (models, demo, tmp, whisper) are ordinary words and are checked as paths only.
    distinctive = [d for d in others if "_" in d or "-" in d]
    distinctive.sort(key=len, reverse=True)
    name_re = re.compile(r"(?<![A-Za-z0-9_])(%s)(?![A-Za-z0-9_])" % "|".join(map(re.escape, distinctive))) \
        if distinctive else None
    path_re = re.compile(r"%s/(%s)(?![A-Za-z0-9_])" % (re.escape(args.workspace.rstrip("/")),
                                                        "|".join(map(re.escape, others)))) if others else None
    for root, dirs, names in os.walk(APP_DIR):
        for d in list(dirs):
            if os.path.islink(os.path.join(root, d)):
                symlinks.append(os.path.relpath(os.path.join(root, d), APP_DIR))
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, APP_DIR)
            if os.path.islink(path):
                symlinks.append("%s -> %s" % (rel, os.readlink(path)))
                continue
            if os.path.getsize(path) > TEXT_LIMIT or not is_text(path):
                continue
            files += 1
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for number, line in enumerate(fh, 1):
                    if path_re and path_re.search(line):
                        path_hits.append("%s:%d: %s" % (rel, number, line.strip()[:140]))
                    elif name_re and name_re.search(line):
                        name_hits.append("%s:%d: %s" % (rel, number, line.strip()[:140]))

    binary = os.path.join(APP_DIR, "vision-local", "build-devkit", "drone-seminar-usb-vision")
    binary_paths = []
    if os.path.isfile(binary):
        out = subprocess.run(["strings", "-n", "6", binary], capture_output=True, text=True).stdout
        binary_paths = sorted({s for s in out.splitlines()
                               if re.search(r"/(workspace|home|media)/", s)})

    manifest = json.load(open(os.path.join(APP_DIR, "config", "model_manifest.json")))["models"]
    models = {k: "%s/%s  (%s)" % (v["root"], v["path"], v["device"]) for k, v in manifest.items()}

    report = {"project": APP_DIR, "text_files_scanned": files, "other_workspace_dirs": others,
              "symlinks": symlinks, "path_references": path_hits, "name_mentions": name_hits,
              "vision_binary_paths": binary_paths, "models": models}
    print("project            : %s (%d text files scanned)" % (APP_DIR, files))
    print("other workspace dirs checked: %d" % len(others))
    print("symlinks           : %d %s" % (len(symlinks), symlinks))
    print("PATH references to another workspace project: %d" % len(path_hits))
    for hit in path_hits:
        print("    " + hit)
    print("NAME mentions of another workspace directory: %d" % len(name_hits))
    for hit in name_hits:
        print("    " + hit)
    print("vision binary absolute paths: %s" % (binary_paths or "none"))
    for key, value in models.items():
        print("model %-14s: %s" % (key, value))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    return 1 if (symlinks or path_hits or binary_paths) else 0


if __name__ == "__main__":
    sys.exit(main())
