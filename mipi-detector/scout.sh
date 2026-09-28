#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${PYNEAT_ENV:-${HOME}/pyneat}/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  echo "pyneat Python not found: ${PYTHON}. Set PYNEAT_ENV or PYTHON." >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
if [ -n "${SCOUT_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${SCOUT_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# Only an explicit whole-group restart returns 75. Wait for the old Python
# process to exit so native device descriptors/allocators cannot survive exec.
scout_child=''
scout_stopping=0
stop_scout() {
  scout_stopping=1
  if [ -n "$scout_child" ]; then
    kill -TERM "$scout_child" 2>/dev/null || true
  fi
}
trap stop_scout INT TERM HUP
while [ "$scout_stopping" -eq 0 ]; do
  "${PYTHON}" -u "${HERE}/scout.py" "$@" &
  scout_child=$!
  scout_status=0
  wait "$scout_child" || scout_status=$?
  if [ "$scout_stopping" -ne 0 ]; then
    # A trapped signal interrupts wait before the child finishes cleanup.
    wait "$scout_child" 2>/dev/null || true
    exit 0
  fi
  scout_child=''
  if [ "$scout_status" -ne 75 ]; then
    exit "$scout_status"
  fi
done
