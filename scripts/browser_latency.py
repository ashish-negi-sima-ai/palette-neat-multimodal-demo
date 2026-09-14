#!/usr/bin/env python3
"""
browser_latency.py - a REAL Chrome renders the demo page; read what it measured.

    runtime/browser/venv/bin/python scripts/browser_latency.py --seconds 30
    runtime/browser/venv/bin/python scripts/browser_latency.py --sync-buffer-ms 0 \
        --overlay off --json runtime/cp/browser/sync0-overlayoff.json

Uses the project-local Chrome for Testing headless shell in runtime/browser/ (no
system install) driven by Playwright. It opens this project's UI exactly as a user
does, lets both panels connect, and samples the page's own instruments:

    window.demoFrameRates()   decoded fps (getStats), shown fps
                              (requestVideoFrameCallback presentedFrames),
                              jitter-buffer delay per frame, overlay draw cost
    window.demoLatency()      per presented frame, using the SAME frame's metadata:
                              receive -> expected display (exact), and DevKit pull ->
                              receive / display (board clock vs this machine's clock)

The DevKit - local clock offset is measured over ssh and applied to the wall-clock
figures, so they are not left depending on two machines agreeing.

IT TAKES THE CHANNELS OVER: vf gives each channel's media to one peer. A browser
already viewing the demo is displaced while this runs.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def board_offset_ms(host):
    # Imported lazily: e2e_latency imports aiortc, which this venv does not have.
    import subprocess
    proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", host,
         "python3 -u -c \"import sys,time\nfor l in sys.stdin: print(time.time_ns(), flush=True)\""],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    time.sleep(1.5)
    res = []
    for _ in range(40):
        a = time.time_ns(); proc.stdin.write("x\n"); proc.stdin.flush()
        b = int(proc.stdout.readline()); c = time.time_ns()
        res.append(((c - a) / 1e6, (b - (a + c) / 2) / 1e6))
        time.sleep(0.02)
    proc.stdin.close(); proc.wait(timeout=5)
    res.sort()
    return statistics.median(o for _, o in res[:10])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seconds", type=float, default=30.0, help="measurement window")
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--sync-buffer-ms", type=int, default=None,
                    help="override render.video_sync_buffer_ms via ?syncBufferMs=")
    ap.add_argument("--overlay", choices=["on", "off"], default="on")
    ap.add_argument("--base", default=None)
    ap.add_argument("--board", default="sima@10.42.0.203")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    with open(os.path.join(APP, "config", "ui_config.json"), encoding="utf-8") as h:
        port = int(json.load(h)["web"]["port"])
    base = args.base or "https://127.0.0.1:%d/" % port
    query = []
    if args.sync_buffer_ms is not None:
        query.append("syncBufferMs=%d" % args.sync_buffer_ms)
    if args.overlay == "off":
        query.append("overlay=off")
    url = base + ("?" + "&".join(query) if query else "")

    offset = board_offset_ms(args.board)
    print("clock: DevKit - local = %+.2f ms" % offset)
    exe = glob.glob(os.path.join(APP, "runtime", "browser", "chrome-headless-shell-*",
                                 "chrome-headless-shell"))
    if not exe:
        sys.exit("no chrome-headless-shell under runtime/browser")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=exe[0], headless=True,
            args=["--ignore-certificate-errors", "--autoplay-policy=no-user-gesture-required",
                  "--disable-background-timer-throttling",
                  "--disable-renderer-backgrounding"])
        page = browser.new_page(ignore_https_errors=True, viewport={"width": 1920, "height": 1080})
        page.goto(url, wait_until="load")
        time.sleep(args.warmup)
        samples = []
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            time.sleep(2.0)
            samples.append(page.evaluate("() => window.demoFrameRates ? window.demoFrameRates() : null"))
        latency = page.evaluate("() => window.demoLatency ? window.demoLatency() : null")
        browser.close()

    result = {"url": url, "offset_ms": offset, "frame_rates": samples, "latency": latency}
    print("url: %s" % url)
    for key in sorted((samples[-1] or {}).get("sources", {})):
        dec, shown, jb, ov = [], [], [], []
        for s in samples:
            v = ((s or {}).get("sources", {}).get(key) or {}).get("video") or {}
            if v.get("decodedFps") is not None: dec.append(v["decodedFps"])
            if v.get("presentedFps") is not None: shown.append(v["presentedFps"])
            if v.get("jitterBufferMs") is not None: jb.append(v["jitterBufferMs"])
            if v.get("overlayMsAvg") is not None: ov.append(v["overlayMsAvg"])
        med = lambda x: round(statistics.median(x), 1) if x else None
        print("source %s: decoded %s fps (min %s)  shown %s fps (min %s)  jitter buffer %s ms  overlay %s ms"
              % (key, med(dec), min(dec) if dec else None, med(shown), min(shown) if shown else None,
                 med(jb), med(ov)))
    for ch, d in sorted(((latency or {}).get("channels") or {}).items()):
        def fmt(name, corr=0.0):
            s = d.get(name)
            if not s:
                return "%s n/a" % name
            return "%s median %.1f p90 %.1f" % (name, s["median"] - corr, s["p90"] - corr)
        # Wall-clock figures are browser-clock minus board-clock; add the measured
        # (board - local) offset to put them on the board clock.
        print("channel %s: %s | %s | %s | %s   (n=%s)" % (
            ch, fmt("browserMs"), fmt("receiveMs", -offset), fmt("displayMs", -offset),
            fmt("boardMs"), d.get("samples")))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as h:
            json.dump(result, h, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
