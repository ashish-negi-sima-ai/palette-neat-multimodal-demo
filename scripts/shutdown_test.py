#!/usr/bin/env python3
"""
shutdown_test.py - does ONE Ctrl+C really stop everything? (issue 0)

    /usr/bin/python3 scripts/shutdown_test.py
    /usr/bin/python3 scripts/shutdown_test.py --with-voice

It runs ./run.sh on a REAL pseudo-terminal, waits until the server is
listening, sends the actual Ctrl+C byte (0x03) to that terminal, and then
checks - from outside - that nothing is left:

    the launcher exited
    no process is running this directory's backend/server.py
    the web port is closed
    the pid file is gone
    the voice server and its ssh tunnel are gone IF run.sh started them

WHY A PTY AND NOT `kill -INT`
-----------------------------
A shell started with `&` from a non-interactive shell has SIGINT and SIGQUIT
set to SIG_IGN, and POSIX says a signal ignored on entry cannot be trapped.
So `./run.sh & kill -INT $!` tests nothing: the signal is discarded before any
trap can see it.  Measured: SigIgn in /proc/<pid>/status reads 0x6 for that
process.  A pty is the only way to reproduce what the operator's keyboard
does.

WHAT IS DELIBERATELY NOT CHECKED
--------------------------------
The camera and YOLO pipeline.  It belongs to the vision process, nothing
in this project signals it, and it is expected to still be running afterwards.
"""

import argparse
import json
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
SERVER = os.path.join(APP, "backend", "server.py")
VOICE = os.path.join(APP, "voice", "local_voice_server.py")


def config_port():
    with open(os.path.join(APP, "config", "ui_config.json"),
              encoding="utf-8") as handle:
        return int(json.load(handle)["web"]["port"])


def pids_running(path):
    """Every pid whose cmdline contains this exact absolute path."""
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % name, "rb") as handle:
                parts = handle.read().split(b"\0")
        except OSError:
            continue
        if path.encode() in parts:
            found.append(int(name))
    return found


def voice_port():
    with open(os.path.join(APP, "config", "ui_config.json"),
              encoding="utf-8") as handle:
        return int(json.load(handle)["voice"]["port"])


def tunnel_pids():
    """The reverse ssh tunnel the voice launcher opens, by its own signature."""
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % name, "rb") as handle:
                line = handle.read().replace(b"\0", b" ")
        except OSError:
            continue
        if (b"ssh" in line and b"-R" in line
                and str(voice_port()).encode() in line):
            found.append(int(name))
    return found


def port_open(port):
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--with-voice", action="store_true",
                        help="let run.sh start the voice server too, and check "
                             "that Ctrl+C stops it and its ssh tunnel")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    port = config_port()
    failures = []

    def check(label, cond, extra=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                               "" if cond else "\n         -> %s" % extra))
        if not cond:
            failures.append(label)

    if pids_running(SERVER):
        sys.exit("a copy of %s is already running; stop it first "
                 "(./run.sh --stop)" % SERVER)
    voice_before = pids_running(VOICE)
    tunnels_before = tunnel_pids()

    argv = ["./run.sh"] + (["--voice"] if args.with_voice else ["--no-voice"])
    print("shutdown_test: %s on a pty (web port %d)" % (" ".join(argv), port))
    if voice_before and args.with_voice:
        print("  note: a voice server was ALREADY running (pid %s). run.sh "
              "leaves\n        a foreign one alone, so it is not expected to "
              "stop." % voice_before)

    primary, secondary = pty.openpty()
    child = subprocess.Popen(argv, cwd=APP, stdin=secondary, stdout=secondary,
                             stderr=secondary, start_new_session=True)
    os.close(secondary)
    os.setsid  # noqa: B018  - documentation: the child owns its own session
    output = []

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([primary], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(primary, 65536)
            except OSError:
                return
            if not chunk:
                return
            text = chunk.decode("utf-8", "replace")
            output.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()

    started = time.time()
    while time.time() - started < args.timeout:
        pump(0.5)
        if "listening on" in "".join(output):
            break
    else:
        child.kill()
        sys.exit("the server never reported 'listening on' within %.0fs"
                 % args.timeout)

    # Give the voice launcher time to finish if it was asked for.
    if args.with_voice:
        deadline = time.time() + 150
        while time.time() < deadline and not pids_running(VOICE):
            pump(1.0)

    voice_started_by_us = [p for p in pids_running(VOICE)
                           if p not in voice_before]
    tunnels_started_by_us = [p for p in tunnel_pids()
                             if p not in tunnels_before]
    server_pids = pids_running(SERVER)
    check("the server is running before the Ctrl+C", bool(server_pids))
    check("the web port is open before the Ctrl+C", port_open(port))
    if args.with_voice:
        print("  voice server started by run.sh: %s" % (voice_started_by_us or
                                                        "none"))

    # The actual keystroke.  ^C on a pty is delivered by the tty driver to the
    # foreground process group, exactly as a keyboard does.
    print("\n--- sending Ctrl+C (0x03) to the terminal ---")
    os.write(primary, b"\x03")

    deadline = time.time() + 60
    while time.time() < deadline and child.poll() is None:
        pump(0.5)
    pump(1.0)

    check("the launcher exited", child.poll() is not None,
          "still running as pid %d" % child.pid)
    if child.poll() is None:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)

    left = pids_running(SERVER)
    check("no copy of backend/server.py is left", not left, left)
    check("the web port %d is closed" % port, not port_open(port))
    pid_files = [name for name in os.listdir(os.path.join(APP, "runtime"))
                 if name.startswith("ui-") and name.endswith(".pid")]
    check("the ui pid file is gone", not pid_files, pid_files)

    if voice_started_by_us:
        still = [p for p in voice_started_by_us if alive(p)]
        check("the voice server run.sh started is stopped", not still, still)
        still_tunnel = [p for p in tunnels_started_by_us if alive(p)]
        check("the ssh tunnel run.sh opened is closed", not still_tunnel,
              still_tunnel)
    elif args.with_voice:
        check("a voice server that was already running is left alone",
              all(alive(p) for p in voice_before), voice_before)

    # Whatever was running before us must still be running: this is the check
    # that the cleanup is narrow rather than broad.
    survivors = [p for p in voice_before if alive(p)]
    check("nothing that predates this test was killed",
          survivors == voice_before, "%s -> %s" % (voice_before, survivors))

    text = "".join(output)
    check("the launcher said what it stopped", "shutting down" in text)
    check("the launcher did not claim to touch the pipeline",
          "never signalled by this script" in text)

    os.close(primary)
    if failures:
        print("\n%d check(s) FAILED" % len(failures))
        return 1
    print("\nall checks passed: one Ctrl+C stopped everything this launcher "
          "started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
