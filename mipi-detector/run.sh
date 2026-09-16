#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${PYNEAT_ENV:-${HOME}/pyneat}/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  echo "pyneat Python not found: ${PYTHON}. Set PYNEAT_ENV or PYTHON." >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON}" -u "${HERE}/main.py" "$@"
