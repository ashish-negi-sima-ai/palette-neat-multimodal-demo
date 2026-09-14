#!/usr/bin/env bash
#
# host/run_host.sh - notebook / SDK-host side check for the Modalix demo.
#
#   host/run_host.sh            check Insight, the DevKit and the demo UI
#
# The notebook runs ONLY display/transport infrastructure: Neat Insight (installed
# with the SDK), WebRTC and the browser. This script starts no AI model and no
# inference - all four models run on the DevKit (./run.sh there).
#
# It reads config/default.env (+ config/local.env) from this same project copy.
set -uo pipefail

HOST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "${HOST_DIR}")"
# shellcheck disable=SC1091
. "${APP_DIR}/config/default.env"
# shellcheck disable=SC1091
[ -f "${APP_DIR}/config/local.env" ] && . "${APP_DIR}/config/local.env"

ok() { printf '  OK       %s\n' "$*"; }
bad() { printf '  PROBLEM  %s\n' "$*"; }

echo "Drone demo - notebook side (no AI inference here)"
echo

INSIGHT_LOCAL="${HOST_INSIGHT_URL:-https://127.0.0.1:${INSIGHT_API_PORT}}"
if curl -sk --max-time 4 "${INSIGHT_LOCAL}/api/health" | grep -q "neat-insight"; then
  ok "Neat Insight is running (${INSIGHT_LOCAL})"
else
  bad "Neat Insight is not answering at ${INSIGHT_LOCAL}"
  echo "           start it the SDK's usual way (neat-insight, port ${INSIGHT_API_PORT})"
fi

stats="$(curl -sk --max-time 4 "${INSIGHT_LOCAL}/api/ingest/stats" 2>/dev/null)"
if [ -n "${stats}" ]; then
  python3 - "${stats}" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
chans = {c["channel"]: c for c in d.get("channels", [])}
for ch in (0, 1):
    c = chans.get(ch)
    if c and c.get("active"):
        print("  OK       video ch%d arriving (%.0f pps), metadata %.1f msg/s"
              % (ch, c["rtp"]["packet_rate_pps"], c["metadata"]["message_rate_mps"]))
    else:
        print("  PROBLEM  no video on ch%d yet - start the demo on the DevKit: ./run.sh" % ch)
PY
fi

if ping -c 1 -W 2 "${DEVKIT_HOST}" >/dev/null 2>&1; then
  ok "DevKit ${DEVKIT_HOST} reachable"
else
  bad "DevKit ${DEVKIT_HOST} not reachable (set DEVKIT_HOST in config/local.env)"
fi

UI="https://${DEVKIT_HOST}:${UI_PORT}"
if curl -sk --max-time 4 -o /dev/null "${UI}/"; then
  ok "Demo UI: ${UI}"
else
  bad "Demo UI not answering at ${UI} (it starts with ./run.sh on the DevKit)"
fi

echo
echo "Open ${UI} in Chrome on this notebook. Browser diagnostics in the console:"
echo "  window.demoFrameRates()   window.demoLatency()"
