#!/usr/bin/env python3
"""
vision_log.py - READ-ONLY view of what the USB cameras and the vision pipeline are saying.

    python3 backend/vision_log.py                  # the interesting lines
    python3 backend/vision_log.py --all            # everything, unclassified
    python3 backend/vision_log.py --json

The camera and YOLO pipeline is this project's vision process (./run.sh starts it), and
its log - runtime/vision-<host>.log - is the only place the camera path's own messages
appear. This UI's log cannot show them: it never sees a frame. So this module READS that
log and the kernel ring buffer and classifies every line, so an operator can tell the
noise from the faults.

NOTHING HERE WRITES ANYWHERE. Both sources are opened read-only:

    runtime/vision-<host>.log   the pipeline's own stdout
    dmesg                       the USB core and the uvcvideo driver

The lines that matter:

    kernel     usb 1-3.3: USB disconnect, device number 4
               usb 1-3.3: reset high-speed USB device number 4 using xhci_hcd
               usb 1-3.1: Not enough bandwidth for altsetting 11
               uvcvideo 1-3.3:1.0: Failed to resubmit video URB (-19).
    pipeline   [ch1 segmentation] [usb] CAMERA DISCONNECTED: ... is no longer on the USB bus
               [ch1 segmentation] [usb] CAMERA RE-ENUMERATED: ... is now /dev/video6 (was /dev/video4)
               [ch1 segmentation] [usb] CAMERA RECONNECTED: ... on /dev/video6
               [ch0 detection] [recovery] CAMERA TIMEOUT / CAMERA RESTART / RUNNER RESTART

A USB camera fault is repaired on that camera alone; the other camera keeps streaming.
"""

import argparse
import json
import os
import re
import subprocess
import time


def _default_vision_log():
    """This project's own vision log (written by ./run.sh), newest per-host file."""
    runtime = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runtime")
    candidates = []
    try:
        for name in os.listdir(runtime):
            if name.startswith("vision-") and name.endswith(".log"):
                path = os.path.join(runtime, name)
                candidates.append((os.path.getmtime(path), path))
    except OSError:
        pass
    return max(candidates)[1] if candidates else os.path.join(runtime, "vision-unknown.log")


VISION_LOG = os.environ.get("VISION_LOG") or _default_vision_log()

# How much of the tail to read.  The log grows to hundreds of KB in an hour, and
# only the recent end is ever interesting.
TAIL_BYTES = 512 * 1024

_STATS_RE = re.compile(r"\[(ch\d) (detection|segmentation)\] \[stats\] (.*)")
_FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


def latest_stats(path=None, max_age_s=15.0):
    """The newest `[stats]` line per channel of the vision log, as numbers.

    {"ch0": {"task": "detection", "yolo_fps": 29.9, "frames": ..., ...}, ...}
    Empty when the log is missing or older than `max_age_s` (vision not running).
    The vision process prints one line per camera every 5 s.
    """
    path = path or VISION_LOG
    try:
        age = time.time() - os.path.getmtime(path)
        if age > max_age_s:
            return {}
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return {}
    out = {}
    for line in tail.splitlines():
        match = _STATS_RE.search(line)
        if not match:
            continue
        entry = {"task": match.group(2)}
        for key, value in _FIELD_RE.findall(match.group(3)):
            try:
                entry[key] = float(value) if "." in value else int(value)
            except ValueError:
                entry[key] = value
        out[match.group(1)] = entry
    return out

# ---------------------------------------------------------------- classification
#
# Each entry is (severity, label, regex).  First match wins, so the benign
# patterns are listed BEFORE the generic ERROR/WARN catch-alls.

BENIGN = [
    ("benign", "the webcam's USB audio interface (not used by the demo)",
     re.compile(r"current rate \d+ is different from the runtime rate")),
    ("benign", "an optional UVC control this camera does not implement",
     re.compile(r"Failed to (set|query) .*UVC (probe )?control")),
    ("benign", "jpegparse cannot parse this camera's APP0 segment (only with DRONE_USB_JPEGPARSE=1)",
     re.compile(r"Failed to parse app0 segment")),
    ("info", "USB camera enumerated",
     re.compile(r"(New USB device found|Found UVC [0-9.]+ device|Product: )")),
    ("info", "exposure_dynamic_framerate switched off for a constant frame rate",
     re.compile(r"exposure_dynamic_framerate")),
]

FAULTS = [
    ("fault", "a USB device left the bus (unplugged, cable or hub problem)",
     re.compile(r"USB disconnect, device number")),
    ("fault", "the USB core reset a device",
     re.compile(r"reset (low|full|high|super|super\+)-speed USB device")),
    ("fault", "the USB bus cannot carry this capture mode next to the other camera",
     re.compile(r"Not enough bandwidth")),
    ("fault", "uvcvideo lost video URBs",
     re.compile(r"uvcvideo.*(Failed to (re)?submit|URB|Non-zero status)")),
    ("fault", "out of memory",
     re.compile(r"(Out of memory|oom-kill|invoked oom-killer)")),
    ("fault", "the pipeline saw a USB camera leave the bus",
     re.compile(r"\[usb\] CAMERA DISCONNECTED")),
    ("event", "a USB camera came back", re.compile(r"\[usb\] CAMERA RECONNECTED|"
                                                  r"camera recovered after")),
    ("event", "Linux gave a reconnected camera another /dev/videoN",
     re.compile(r"CAMERA RE-ENUMERATED")),
    ("fault", "a V4L2 capture error reached the pipeline",
     re.compile(r"CAMERA PULL ERROR")),
    ("fault", "the pipeline saw no camera frames for this long",
     re.compile(r"CAMERA TIMEOUT: stalled")),
    ("fault", "the pipeline rebuilt a camera graph (this camera only)",
     re.compile(r"CAMERA RESTART")),
    ("fault", "the pipeline rebuilt a model Runner",
     re.compile(r"RUNNER (RESTART|TIMEOUT)")),
    ("fault", "a camera graph could not be rebuilt yet",
     re.compile(r"stream graph restore (failed|after the pause failed)")),
    ("event", "a camera graph was stopped or restored",
     re.compile(r"stream graph (STOP|STOPPED|RESTORED|restored)")),
    ("event", "vision pause taken for maintenance or a voice window",
     re.compile(r"controlled vision pause|requesting a vision pause")),
]


def classify(line):
    """(severity, why) for one log line.  severity: fault/event/benign/info/other."""
    for severity, why, pattern in BENIGN:
        if pattern.search(line):
            return severity, why
    for severity, why, pattern in FAULTS:
        if pattern.search(line):
            return severity, why
    if re.search(r"\bERROR\b", line):
        return "other", "unclassified ERROR"
    if re.search(r"\bWARN\b", line):
        return "other", "unclassified WARN"
    return "other", None


# ---------------------------------------------------------------- the sources

def read_vision_log(path=VISION_LOG, tail_bytes=TAIL_BYTES):
    """The tail of the pipeline's own log, or a reason it is not readable."""
    if not os.path.isfile(path):
        return [], ("%s does not exist. The camera + YOLO pipeline is this project's "
                    "vision process; its log exists once ./run.sh has started it."
                    % path)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > tail_bytes:
                handle.seek(size - tail_bytes)
                handle.readline()          # drop the partial first line
            body = handle.read().decode("utf-8", "replace")
    except OSError as exc:
        return [], "cannot read %s: %s" % (path, exc)
    return body.splitlines(), None


def read_kernel(lines=4000):
    """USB core / uvcvideo / OOM messages from the kernel ring buffer, if readable here."""
    try:
        result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True,
                                timeout=10)
        if result.returncode != 0:
            result = subprocess.run(["dmesg"], capture_output=True, text=True,
                                    timeout=10)
        if result.returncode != 0:
            return [], ("dmesg is not readable here (%s). The USB and uvcvideo "
                        "messages only exist on the machine with the cameras."
                        % (result.stderr or "permission denied").strip()[:80])
    except (OSError, subprocess.SubprocessError) as exc:
        return [], "dmesg is unavailable here: %s" % exc
    keep = re.compile(r"(usb \d+-[0-9.]+|uvcvideo|xhci|bandwidth|oom|Out of memory)", re.I)
    out = [line for line in result.stdout.splitlines()[-lines:] if keep.search(line)]
    return out, None


# ------------------------------------------------------------------ the report

_SELECTED = re.compile(r"USB Camera (\d) identity\s*: (\S+)\s+\(\"([^\"]*)\", usb ([0-9.-]+)")
_STREAM = {"0": ("ch0 detection", "LEFT"), "1": ("ch1 segmentation", "RIGHT")}


def usb_health(kernel_lines, vision_lines):
    """Per-USB-port fault tallies, attributed to the demo camera on that port.

    The camera <-> port mapping is read from the vision process's own startup block
    ("USB Camera N identity : ... usb 1-3.3 ..."), so nothing here is guessed.
    """
    cameras = {}
    for line in vision_lines:
        match = _SELECTED.search(line)
        if match:
            index, identity, card, path = match.groups()
            stream, panel = _STREAM.get(index, ("?", "?"))
            cameras[path] = {"camera": index, "identity": identity, "sensor": card,
                             "stream": stream, "panel": panel}
    out = {}

    def entry(path):
        return out.setdefault(path, {"disconnects": 0, "resets": 0,
                                     "bandwidth_refusals": 0, "urb_errors": 0})

    patterns = [
        ("disconnects", re.compile(r"usb (\d+-[0-9.]+): USB disconnect")),
        ("resets", re.compile(r"usb (\d+-[0-9.]+): reset \S+-speed USB device")),
        ("bandwidth_refusals", re.compile(r"usb (\d+-[0-9.]+): Not enough bandwidth for altsetting")),
        ("urb_errors", re.compile(r"uvcvideo (\d+-[0-9.]+):[0-9.]+: .*(URB|Non-zero status)")),
    ]
    for line in kernel_lines:
        for key, pattern in patterns:
            match = pattern.search(line)
            if match:
                entry(match.group(1))[key] += 1
                break
    for path in cameras:
        entry(path)
    for path, e in out.items():
        e.update(cameras.get(path, {"sensor": "(not a demo camera)", "stream": "-"}))
        e["usb_path"] = path
        problems = [k for k in ("disconnects", "resets", "urb_errors") if e[k]]
        if problems:
            e["verdict"] = ("since boot: %s. Check this camera's cable, hub and power."
                            % ", ".join("%d %s" % (e[k], k.replace("_", " ")) for k in problems))
        else:
            e["verdict"] = "no disconnect, reset or URB error since boot"
        if e["bandwidth_refusals"]:
            e["verdict"] += (" %d bandwidth refusal(s) - expected while run.sh probes modes "
                             "the shared USB bus cannot carry for two cameras."
                             % e["bandwidth_refusals"])
    return out


def report(vision_path=VISION_LOG, limit=60, include_benign=False):
    vision_lines, vision_error = read_vision_log(vision_path)
    kernel_lines, kernel_error = read_kernel()

    links = usb_health(kernel_lines, vision_lines)
    classified = []
    counts = {}
    for line in vision_lines + kernel_lines:
        if not line.strip():
            continue
        severity, why = classify(line)
        counts[severity] = counts.get(severity, 0) + 1
        if severity in ("fault", "event") or include_benign:
            classified.append({"severity": severity, "why": why,
                               "line": line.rstrip()[:400]})

    return {
        "at": time.time(),
        "vision_log": vision_path,
        "vision_log_error": vision_error,
        "kernel_error": kernel_error,
        "counts": counts,
        "links": links,
        "vdma_watchdog_giveups": {},
        "lines": classified[-limit:],
        "benign_note": ("A USB webcam also exposes an audio interface; its kernel "
                        "'current rate ... is different from the runtime rate' lines and "
                        "unsupported optional UVC controls are printed on every start and "
                        "are not faults."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--log", default=VISION_LOG)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--all", action="store_true",
                        help="include the benign noise")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    data = report(args.log, limit=args.limit, include_benign=args.all)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    print("camera messages  (%s)" % data["vision_log"])
    for key in ("vision_log_error", "kernel_error"):
        if data[key]:
            print("  NOTE: %s" % data[key])
    print("")
    if data["links"]:
        print("USB camera health --------------------------------------------")
        for path, entry in sorted(data["links"].items()):
            print("  usb %s  %s  -> %s" % (path, entry.get("sensor", "?"),
                                          entry.get("stream", "?")))
            print("      disconnects %d  resets %d  URB errors %d  bandwidth refusals %d"
                  % (entry["disconnects"], entry["resets"], entry["urb_errors"],
                     entry["bandwidth_refusals"]))
            print("      verdict: %s" % entry["verdict"])
        print("")
    print("counts by severity: %s" % json.dumps(data["counts"]))
    print("  (%s)" % data["benign_note"])
    print("")
    print("recent camera faults and events ------------------------------")
    for entry in data["lines"]:
        print("  [%s] %s" % (entry["severity"], entry["line"]))
        if entry["why"]:
            print("        %s" % entry["why"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
