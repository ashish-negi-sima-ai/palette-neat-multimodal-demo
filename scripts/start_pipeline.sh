#!/usr/bin/env bash
#
# start_pipeline.sh - start THIS project's vision process on the Modalix DevKit.
#
#   scripts/start_pipeline.sh                           (normally started by ./run.sh)
#   scripts/start_pipeline.sh --selection <file>        use run.sh's preflight camera selection
#   scripts/start_pipeline.sh --no-preflight
#
# It runs exactly one program: vision-local/build-devkit/drone-seminar-usb-vision
#
#   USB camera 0 -> MJPEG -> neatdecoder NV12 -> H.264 -> udp INSIGHT_HOST:VIDEO_PORT_BASE+0
#                -> YOLO26m detection -> metadata udp INSIGHT_HOST:METADATA_PORT_BASE+0
#   USB camera 1 -> MJPEG -> neatdecoder NV12 -> H.264 -> udp INSIGHT_HOST:VIDEO_PORT_BASE+1
#                -> YOLO26m-seg        -> metadata udp INSIGHT_HOST:METADATA_PORT_BASE+1
#
# Cameras: with --selection (written by run.sh's preflight probe) the two cameras are
# passed by their stable USB identity and the probed capture mode is used as is.
# Without it the binary discovers and probes by itself from USB_CAMERA* in
# config/default.env (+ config/local.env).
#
# There is NO fallback. If the binary, its configuration or a model artifact is
# missing, this fails and says what is missing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "${HERE}")"

die() { echo "[start_pipeline] ERROR: $*" >&2; exit 1; }
say() { echo "[start_pipeline] $*"; }

# shellcheck disable=SC1091
. "${APP_DIR}/config/default.env"
# shellcheck disable=SC1091
[ -f "${APP_DIR}/config/local.env" ] && . "${APP_DIR}/config/local.env"

VISION_BIN="${APP_DIR}/vision-local/build-devkit/drone-seminar-usb-vision"
VISION_CONFIG="${APP_DIR}/config/vision.yaml"
PREFLIGHT=1
SELECTION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --insight-host) INSIGHT_HOST="${2:?--insight-host needs a value}"; shift 2 ;;
    --selection) SELECTION="${2:?--selection needs a file}"; shift 2 ;;
    --no-preflight) PREFLIGHT=0; shift ;;
    --help|-h) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[ -x "${VISION_BIN}" ] || die "missing ${VISION_BIN}
       build it in the SDK container:  vision-local/build.sh"
[ -f "${VISION_CONFIG}" ] || die "missing ${VISION_CONFIG}"

# Model artifacts, from the manifest only. Missing = stop; nothing is substituted.
MODEL_PATHS="$(python3 "${APP_DIR}/scripts/check_models.py" --only yolo26m,yolo26m_seg --print-paths)" \
  || die "required YOLO model artifacts are missing (see above)"
DET_MODEL="$(echo "${MODEL_PATHS}" | sed -n 's/^yolo26m=//p')"
SEG_MODEL="$(echo "${MODEL_PATHS}" | sed -n 's/^yolo26m_seg=//p')"

CAM_ARGS=()
if [ -n "${SELECTION}" ]; then
  [ -f "${SELECTION}" ] || die "camera selection file ${SELECTION} does not exist"
  # shellcheck disable=SC1090
  . "${SELECTION}"
  [ -n "${USB_SELECTED_CAMERA0_IDENTITY:-}" ] && [ -n "${USB_SELECTED_CAMERA1_IDENTITY:-}" ] \
    && [ -n "${USB_SELECTED_MODE:-}" ] || die "incomplete camera selection in ${SELECTION}"
  CAM_ARGS=(--camera0 "identity:${USB_SELECTED_CAMERA0_IDENTITY}"
            --camera1 "identity:${USB_SELECTED_CAMERA1_IDENTITY}"
            --usb-mode "${USB_SELECTED_MODE}" --skip-mode-probe)
  say "cameras       : camera0 ${USB_SELECTED_CAMERA0_IDENTITY}, camera1 ${USB_SELECTED_CAMERA1_IDENTITY}, ${USB_SELECTED_MODE} (run.sh preflight)"
else
  [ -n "${USB_CAMERA0_DEVICE}" ] && CAM_ARGS+=(--camera0 "${USB_CAMERA0_DEVICE}")
  [ -n "${USB_CAMERA1_DEVICE}" ] && CAM_ARGS+=(--camera1 "${USB_CAMERA1_DEVICE}")
  [ -n "${USB_CAMERA_FILTER}" ] && CAM_ARGS+=(--camera-filter "${USB_CAMERA_FILTER}")
  [ -n "${USB_CAMERA_MODE}" ] && CAM_ARGS+=(--usb-mode "${USB_CAMERA_MODE}")
  CAM_ARGS+=(--usb-format "${USB_CAMERA_FORMAT}" --usb-modes "${USB_CAMERA_MODES}")
  say "cameras       : discovered and probed by the vision binary (USB_CAMERA* settings)"
fi

if [ "${PREFLIGHT}" -eq 1 ]; then
  say "preflight: USB capture cameras"
  "${VISION_BIN}" --list-cameras || say "  WARNING: fewer than two USB capture cameras listed"
fi

# The parallel graph takes the SDK default async-queue behaviour.
unset SIMA_ASYNC_QUEUE2_DEPTH SIMA_QUEUE_LEAKY_DOWNSTREAM

export TMPDIR="${APP_DIR}/runtime/tmp"
export XDG_CACHE_HOME="${APP_DIR}/runtime/cache"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

say "vision binary : ${VISION_BIN}"
say "config        : ${VISION_CONFIG}"
say "detection     : ${DET_MODEL}"
say "segmentation  : ${SEG_MODEL}"
say "voice lease   : pause behaviour ${VISION_PAUSE_BEHAVIOUR} (what the cameras do while voice holds the MLA)"
say "outputs       : video udp ${INSIGHT_HOST}:${VIDEO_PORT_BASE}+ch  metadata udp ${INSIGHT_HOST}:${METADATA_PORT_BASE}+ch"
for v in DRONE_VIDEO_INGRESS DRONE_PRESET DRONE_QUEUE_DEPTH DRONE_FRAMES_OUTPUT DRONE_OUTPUT_MEMORY DRONE_STATS_S DRONE_RUN_EXPORT_DIR DRONE_USB_JPEGPARSE DRONE_USB_DECODER_NEXT DRONE_SNAPSHOT_DIR DRONE_SNAPSHOT_EVERY_S; do
  [ -n "${!v:-}" ] && say "  ${v}=${!v}"
done

# DIAGNOSTIC ONLY: extra vision flags for measurement runs, e.g.
#   VISION_EXTRA_ARGS="--no-inference"   (camera + video, no YOLO Model/Runner)
# Empty in normal operation.
read -r -a EXTRA_ARGS <<< "${VISION_EXTRA_ARGS:-}"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && say "DIAGNOSTIC extra vision args: ${EXTRA_ARGS[*]}"

# Relative paths in vision.yaml (the labels file) resolve against the project root.
cd "${APP_DIR}" || die "cannot enter ${APP_DIR}"
exec "${VISION_BIN}" \
  --config "${VISION_CONFIG}" \
  --host "${INSIGHT_HOST}" \
  "${CAM_ARGS[@]}" \
  --det-model "${DET_MODEL}" \
  --seg-model "${SEG_MODEL}" \
  --labels "${APP_DIR}/assets/coco_labels.txt" \
  --arbiter-port "${ARBITER_PORT}" \
  --pause-behaviour "${VISION_PAUSE_BEHAVIOUR}" \
  --port-base "${VIDEO_PORT_BASE}" \
  --metadata-port-base "${METADATA_PORT_BASE}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
