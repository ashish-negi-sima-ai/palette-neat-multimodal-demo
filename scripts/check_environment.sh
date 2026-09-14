#!/usr/bin/env bash
#
# check_environment.sh - is this DevKit (and this project copy) able to run the demo?
#
#   scripts/check_environment.sh            (on the Modalix DevKit)
#   ./run.sh --check                        (same, from the DevKit or the SDK host)
#
# Prints PASS / WARNING / FAIL per check and exits non-zero on any FAIL.
# It only READS: nothing is installed, started, stopped or modified. (It lists the USB
# cameras; it does not stream them - ./run.sh's preflight probes the capture mode.)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "${HERE}")"
# shellcheck disable=SC1091
. "${APP_DIR}/config/default.env"
# shellcheck disable=SC1091
[ -f "${APP_DIR}/config/local.env" ] && . "${APP_DIR}/config/local.env"
export XDG_CACHE_HOME="${APP_DIR}/runtime/cache" TMPDIR="${APP_DIR}/runtime/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${TMPDIR}"

FAILS=0
WARNS=0
pass() { printf '  PASS     %s\n' "$*"; }
warn() { printf '  WARNING  %s\n' "$*"; WARNS=$((WARNS + 1)); }
fail() { printf '  FAIL     %s\n' "$*"; FAILS=$((FAILS + 1)); }

echo "Palette NEAT SDK Demo environment check"
echo "  project root : ${APP_DIR}"
echo "  MODEL_ROOT   : ${MODEL_ROOT}"
echo

echo "[platform]"
arch="$(uname -m)"
[ "${arch}" = "aarch64" ] && pass "architecture aarch64" || fail "architecture ${arch} (run this on the Modalix DevKit)"
if grep -qi modalix /proc/device-tree/model 2>/dev/null; then
  pass "board: $(tr -d '\0' < /proc/device-tree/model)"
else
  warn "board model does not say Modalix ($(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown))"
fi
if [ -r /etc/buildinfo ]; then
  pass "SiMa build: $(grep -E 'SIMA_BUILD_VERSION' /etc/buildinfo | cut -d= -f2 | tr -d ' ')"
else
  warn "no /etc/buildinfo (SDK/BSP version unknown)"
fi
mem_total="$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)"
[ "${mem_total}" -ge 5000 ] && pass "RAM ${mem_total} MB" || warn "RAM ${mem_total} MB (measured demo needs ~2.5 GB with all models)"

echo "[NEAT / GStreamer]"
if ls /usr/lib/aarch64-linux-gnu/libsima_neat.so* >/dev/null 2>&1 || ls /usr/lib/libsima_neat.so* >/dev/null 2>&1; then
  pass "libsima_neat present"
else
  fail "libsima_neat not found"
fi
for el in v4l2src neatdecoder neatencoder h264parse rtph264pay udpsink appsrc appsink queue; do
  if gst-inspect-1.0 "${el}" >/dev/null 2>&1; then pass "GStreamer element ${el}"; else fail "GStreamer element ${el} missing"; fi
done

echo "[MLA / GenAI runtime]"
if pgrep -x mlashmcomplex >/dev/null 2>&1; then pass "MLA service mlashmcomplex running"; else fail "MLA service mlashmcomplex not running"; fi
[ -e /dev/simaai-mem ] && pass "/dev/simaai-mem" || warn "/dev/simaai-mem not present"
if [ -x "${PYNEAT_ENV}/bin/python3" ]; then
  if out="$("${PYNEAT_ENV}/bin/python3" -c 'import pyneat as n; g=n.genai; assert hasattr(g,"ASRModel") and hasattr(g,"GenAIModel"); print(getattr(n,"__version__","?"))' 2>&1)"; then
    pass "pyneat ${out} with genai.ASRModel + genai.GenAIModel (${PYNEAT_ENV})"
  else
    fail "pyneat import failed in ${PYNEAT_ENV}: ${out}"
  fi
else
  fail "PYNEAT_ENV ${PYNEAT_ENV}/bin/python3 not found (NEAT GenAI Python runtime)"
fi

VISION_BIN="${APP_DIR}/vision-local/build-devkit/drone-seminar-usb-vision"
echo "[USB cameras]"
if [ -x "${VISION_BIN}" ]; then
  listing="$("${VISION_BIN}" --list-cameras 2>&1)"
  count="$(echo "${listing}" | grep -cE '^\s+\[[0-9]+\] /dev/video')"
  echo "${listing}" | sed 's/^/           /'
  if [ "${count}" -ge 2 ]; then
    pass "${count} USB UVC capture camera(s) found (two are used)"
  else
    fail "${count} USB UVC capture camera(s) found; the demo needs two"
  fi
  [ "${count}" -gt 2 ] && [ -z "${USB_CAMERA0_DEVICE}${USB_CAMERA1_DEVICE}${USB_CAMERA_FILTER}" ] && \
    warn "more than two cameras and no USB_CAMERA0_DEVICE / USB_CAMERA1_DEVICE / USB_CAMERA_FILTER: run.sh will refuse to guess"
else
  fail "cannot list USB cameras: vision binary missing"
fi
if [ -d /sys/module/uvcvideo ]; then pass "uvcvideo driver loaded"; else fail "uvcvideo driver not loaded"; fi

echo "[audio input]"
if [ "${AUDIO_SOURCE}" = "browser" ]; then
  pass "AUDIO_SOURCE=browser (the page records 16 kHz mono and uploads it; no board sound card needed)"
else
  grep -q '[0-9]' /proc/asound/cards 2>/dev/null && pass "board sound card present" || fail "AUDIO_SOURCE=${AUDIO_SOURCE} but no sound card"
fi

echo "[models: config/model_manifest.json]"
if [ -d "${MODEL_ROOT}" ]; then pass "MODEL_ROOT ${MODEL_ROOT}"; else fail "MODEL_ROOT ${MODEL_ROOT} does not exist"; fi
if out="$(MODEL_ROOT="${MODEL_ROOT}" python3 "${APP_DIR}/scripts/check_models.py" --machine devkit 2>&1)"; then :; else FAILS=$((FAILS + 1)); fi
while read -r status rest; do
  [ -z "${status}" ] && continue
  case "${status}" in PASS) pass "${rest}" ;; *) fail "${rest}" ;; esac
done <<< "${out}"
if python3 - "${APP_DIR}" <<'PYC'
import json, sys
m = json.load(open(sys.argv[1] + "/config/model_manifest.json"))["models"]
ok = set(m) == {"yolo26m", "yolo26m_seg", "whisper_medium", "qwen3_0_6b"} \
    and all(e.get("device") == "Modalix MLA" for e in m.values()) \
    and m["yolo26m"]["root"] == "PROJECT" and m["yolo26m_seg"]["root"] == "PROJECT"
sys.exit(0 if ok else 1)
PYC
then pass "manifest: YOLO26m + YOLO26m-seg in this project, Whisper-medium + Qwen3-0.6B under MODEL_ROOT, all on Modalix MLA"; else fail "manifest does not list the four MLA models as expected"; fi

echo "[project binaries and files]"
if [ -x "${VISION_BIN}" ]; then
  if grep -a -q "fixes active: usb-local-camera-restart usb-rediscovery" "${VISION_BIN}"; then
    pass "vision binary carries the USB camera recovery"
  else
    fail "vision binary lacks the USB recovery marker (rebuild: vision-local/build.sh)"
  fi
  pass "vision binary ${VISION_BIN#${APP_DIR}/}"
  if ldd "${VISION_BIN}" 2>/dev/null | grep -q "not found"; then
    fail "vision binary has unresolved libraries: $(ldd "${VISION_BIN}" | grep 'not found' | tr -s ' ' | tr '\n' ';')"
  else
    pass "vision binary libraries resolve"
  fi
else
  fail "vision binary missing - build in the SDK container: vision-local/build.sh"
fi
for f in config/vision.yaml assets/coco_labels.txt backend/server.py backend/command_config.json \
         backend/command_processor.py voice/board_voice_server.py voice/command_prompt.txt web/index.html; do
  [ -f "${APP_DIR}/${f}" ] && pass "${f}" || fail "${f} missing"
done

echo "[network / ports]"
for port in "${UI_PORT}" "${VOICE_PORT}" "${ARBITER_PORT}"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    warn "tcp ${port} already in use (a demo already running?)"
  else
    pass "tcp ${port} free"
  fi
done
if curl -sk --max-time 4 -o /dev/null "https://${INSIGHT_HOST}:${INSIGHT_API_PORT}/api/health"; then
  pass "Insight reachable at https://${INSIGHT_HOST}:${INSIGHT_API_PORT}"
else
  warn "Insight not reachable at https://${INSIGHT_HOST}:${INSIGHT_API_PORT} (start Insight on the notebook)"
fi

echo
echo "summary: ${FAILS} FAIL, ${WARNS} WARNING"
[ "${FAILS}" -eq 0 ]
