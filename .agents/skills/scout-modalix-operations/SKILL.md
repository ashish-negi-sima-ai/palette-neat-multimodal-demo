---
name: scout-modalix-operations
description: "Run, stop, restart, validate, or troubleshoot this repository's SCOUT application and MIPI detector on a Modalix SoM or DVT board. Use for camera discovery, Gemma loading, runtime compatibility, FPS, USB recovery, or Neat Insight stream and overlay failures."
---

# SCOUT on Modalix

Read [SCOUT.md](../../../mipi-detector/SCOUT.md) for the launch command matching
the target board. For the plain detector, read its
[README](../../../mipi-detector/README.md). The root `run.sh` controls the
separate USB/voice demo; it does not manage SCOUT.

## Establish the target

Use the latest host and user supplied in the session. Addresses in documentation
are historical examples, and DHCP has changed them repeatedly. Identify the
board, installed runtime, camera names and current SCOUT process before choosing
settings. A diagnostic request starts with read-only inspection; starting,
stopping or restarting follows the user's requested scope and existing
authorization. A request for explanations or commands alone does not require
changing the board's state.

Useful checks on the board:

```bash
tr '\0' '\n' < /proc/device-tree/model
cam -l
ls -l /dev/v4l/by-id/
ps -eo pid,etime,args | grep '[m]ipi-detector/scout.py'
ss -ltnp 'sport = :8022'
```

`cam -l` demonstrates sensor registration; capture still needs validation.
Many `/dev/video*` nodes are ISP/codec endpoints, not separate connected cameras.
USB `auto` selects the sole stable `*-video-index0` identity; use an explicit
identity when multiple cameras are present. Read
[runtime troubleshooting](references/runtime-troubleshooting.md) when camera
configuration, compatibility, FPS or stream delivery is the problem.

## Run or restart the requested application

- Confirm the checkout and compiled YOLO archive exist on the board. SCOUT's
  Gemma directory must contain `devkit/vlm_config.json` and the nonempty vision
  and language ELFs referenced by it; `scout_vlm.validate_model` checks this.
- The launcher uses `~/pyneat/bin/python` unless `PYTHON` or `PYNEAT_ENV`
  overrides it. `--host` is the Insight machine, not the camera board.
  Verify reachable Insight ports against the SDK's `neat --json` mapping.
- Use the appropriate launch command in `SCOUT.md`. Add `--usb-camera auto`
  for the USB watch view; omit it for MIPI only. Initial Gemma loading can take
  roughly two minutes over the validated shared filesystem. Follow readiness
  and error logs before deciding that the startup failed.
- To stop, use Ctrl+C in the owning terminal or SIGTERM to the confirmed SCOUT
  parent PID. Wait for shutdown, camera release and port release before
  restarting. An existing `live.pid` must be checked against `/proc/<pid>/cmdline`;
  the launcher itself does not create or maintain that PID file. Prefer graceful
  cleanup of the VLM worker and graph over killing every Python process.
- A detached launch needs an explicit log destination and recorded parent PID.
  Existing sessions have used `~/.cache/neat-scout/live.log` and `live.pid`;
  establish the actual launch configuration rather than assuming those files exist.
  Share the console URL and stop command when leaving a requested app running.

## Verify the outcome

The console is normally `https://<board>:8022`. Inspect `/api/state` for each
camera's status, frame count, FPS, observation age, errors and VLM readiness.
The board's self-signed certificate may require `curl -k` for diagnostics.

For a launch or pipeline fix, observe increasing frames and browser video on
each configured view. For a VLM/lifecycle change, inspect an actual visible
subject or USB region, open its evidence, and confirm both streams resume.
Check overlay correlation with `scout.source.stats()` and, when configured,
`scout.usbSource.stats()` in the browser. API status or metadata receipt alone
does not prove that video and overlays match.

Only exercise USB disconnect/reconnect when relevant to the requested change.
For a software driver-disconnect test, identify the exact USB camera interface
and guarantee restoration in a `finally` block. Verify that USB occupancy becomes
unknown, MIPI keeps advancing, and USB video plus overlays return.

Normal shutdown writes `~/.cache/neat-scout/last-run.json` (or `--runtime-dir`)
with event metadata. Evidence images are held in RAM and disappear at shutdown.
Report model generation latency separately from total capture-pause/recovery
time, and distinguish measured FPS from requested FPS.
