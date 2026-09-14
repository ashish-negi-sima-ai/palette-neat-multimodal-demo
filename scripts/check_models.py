#!/usr/bin/env python3
"""
check_models.py - verify every model artifact listed in config/model_manifest.json.

    scripts/check_models.py [--machine devkit|host] [--only key1,key2] [--print-paths] [--sha256]

Each manifest entry states the MACHINE that uses it and the ROOT variable it lives under
(MODEL_ROOT on the DevKit, HOST_MODEL_ROOT on the notebook). Root values come from the
environment, else from config/default.env (+ config/local.env), resolved by bash exactly
as the launchers resolve them. A missing artifact is a FAIL (exit 1). Nothing is
downloaded or substituted. --sha256 additionally verifies recorded SHA256 values.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(APP, "config", "model_manifest.json")


def resolved_env():
    script = ('set -a; . "%s/config/default.env"; [ -f "%s/config/local.env" ] && . "%s/config/local.env"; '
              'env -0' % (APP, APP, APP))
    out = subprocess.run(["bash", "-c", script], capture_output=True, check=True).stdout
    return dict(item.split("=", 1) for item in out.decode("utf-8", "replace").split("\0") if "=" in item)


def root_value(name, env=None):
    if name == "PROJECT":
        return APP
    if os.environ.get(name):
        return os.environ[name]
    return (env or resolved_env()).get(name, "")


def model_root():
    """MODEL_ROOT (DevKit model storage)."""
    return root_value("MODEL_ROOT")


def artifact_path(entry, env=None):
    return os.path.join(root_value(entry.get("root", "MODEL_ROOT"), env), entry["path"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--machine", choices=["devkit", "host"], default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--print-paths", action="store_true")
    ap.add_argument("--sha256", action="store_true")
    args = ap.parse_args()
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    env = resolved_env()
    wanted = [k for k in args.only.split(",") if k]
    failures = 0
    for key, entry in manifest["models"].items():
        if wanted and key not in wanted:
            continue
        if args.machine and entry.get("machine") != args.machine:
            continue
        path = artifact_path(entry, env)
        ok = os.path.isfile(path) if entry.get("kind") == "file" else os.path.isdir(path)
        missing = [f for f in entry.get("required_files", []) if not os.path.exists(os.path.join(path, f))]
        status = "PASS" if ok and not missing else "FAIL"
        detail = ""
        if ok and not missing and args.sha256 and entry.get("sha256"):
            recorded = entry["sha256"]
            if isinstance(recorded, str):
                bad = [] if sha256(path) == recorded else [os.path.basename(path)]
            else:
                bad = [f for f, h in recorded.items() if sha256(os.path.join(path, f)) != h]
            if bad:
                status, detail = "FAIL", " (sha256 mismatch: %s)" % ", ".join(bad)
            else:
                detail = " (sha256 verified)"
        if missing:
            detail = " (missing: %s)" % ", ".join(missing)
        if status != "PASS":
            failures += 1
        out = sys.stderr if args.print_paths else sys.stdout
        print("%s  %-14s %s  [%s on %s]%s" % (status, key, path, entry.get("runtime", "?"), entry.get("device", "?"), detail), file=out)
        if args.print_paths and status == "PASS":
            print("%s=%s" % (key, path))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
