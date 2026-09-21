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
exec "${PYTHON}" -u "${HERE}/scout.py" "$@"
