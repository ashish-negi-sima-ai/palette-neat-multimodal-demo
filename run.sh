#!/usr/bin/env bash
#
# run.sh - the ONE entry point for the Palette NEAT SDK Demo (dual USB cameras).
#
#   cd <this project directory>       (any location; paths resolve from this script)
#   ./run.sh              preflight, start everything, Ctrl+C stops everything it started
#   ./run.sh --stop-all   stop every project-owned process (UI, vision, voice)
#   ./run.sh --status     what is running
#   ./run.sh --check      read-only environment check (scripts/check_environment.sh)
#
# ALL AI RUNS ON THE MODALIX DEVKIT. The SDK host only displays (Insight, browser).
#
# WHAT ONE ./run.sh DOES
#
#   0. PREFLIGHT (nothing is started if any check fails):
#        no stale instance of this project, vision binary + libraries + GStreamer elements,
#        YOLO archives (in this project), Whisper-medium + Qwen3-0.6B (under MODEL_ROOT),
#        pyneat, free ports, and the USB cameras: two UVC capture nodes are discovered,
#        and a capture mode that BOTH cameras can stream at the same time is probed and
#        printed (device, format, resolution, fps per camera).
#   1. the vision process (vision-local/build-devkit/drone-seminar-usb-vision):
#        USB camera 0 -> MJPEG -> NV12 -> H.264 (udp VIDEO_PORT_BASE+0) + YOLO26m    (meta +0)
#        USB camera 1 -> MJPEG -> NV12 -> H.264 (udp VIDEO_PORT_BASE+1) + YOLO26m-seg (meta +1)
#      On the DevKit it runs locally. From the SDK host it is started on the DevKit
#      over ssh ($DEVKIT_USER@$DEVKIT_HOST from config/default.env).
#      The launcher waits until BOTH cameras report frames, or reports the failure.
#   2. Insight: checked, never started or stopped (it is the SDK's service).
#   3. the voice runtime (voice/board_voice_server.py), ON THE DEVKIT:
#        Whisper-medium (speech -> text) and Qwen3-0.6B (only for commands the
#        deterministic parser cannot decide), both on the MLA through pyneat. It
#        takes the vision process's MLA lease around each inference. Voice and
#        typed commands share one command path (backend/command_processor.py).
#   4. the Demo UI web server (backend/server.py), which is told the capture mode
#      the preflight selected (runtime/usb-selection-*.env).
#   5. a summary that names the four AI workloads and confirms each is on the MLA.
#
# Options:
#   --no-vision        do not start/connect the vision pipeline (UI-only testing)
#   --no-voice         do not start the on-device voice runtime
#   --background       start everything and return
#   --stop             stop only the UI server
#   --stop-all         stop the UI server, the vision pipeline and the voice server
#                      that this project owns (pid files verified against /proc)
#   --status           report all of them
#   --check            scripts/check_environment.sh (on the DevKit)
#   --selftest         offline checks (scripts/selftest.py)
#   everything else    forwarded to backend/server.py (--port, --insight-host, ...)
#
# PROCESS OWNERSHIP
#
# Every process this launcher starts is recorded in runtime/*.pid and is only ever
# signalled after /proc/<pid>/exe (vision) or /proc/<pid>/cmdline (UI, voice) proves it
# is THIS project's program. No pkill, no killall, no name pattern. Insight and other
# projects' processes are never touched. Runtime files are per host, because
# the project tree may be shared (NFS) by the SDK host and the DevKit.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTTAG="$(hostname 2>/dev/null | tr -dc 'A-Za-z0-9._-' | cut -c1-40)"
HOSTTAG="${HOSTTAG:-unknown}"
PID_FILE="${HERE}/runtime/ui-${HOSTTAG}.pid"
LOG_FILE="${HERE}/runtime/ui-${HOSTTAG}.log"
SERVER="${HERE}/backend/server.py"
CONFIG="${HERE}/config/ui_config.json"

# Site configuration: config/default.env, then config/local.env if present.
# shellcheck disable=SC1091
. "${HERE}/config/default.env"
# shellcheck disable=SC1091
[ -f "${HERE}/config/local.env" ] && . "${HERE}/config/local.env"
export MODEL_ROOT DEVKIT_HOST DEVKIT_USER INSIGHT_HOST VIDEO_PORT_BASE METADATA_PORT_BASE \
       UI_BIND UI_PORT SYNC_BUFFER_MS AUDIO_SOURCE VOICE_PORT \
       PYNEAT_ENV ARBITER_PORT VISION_PAUSE_BEHAVIOUR VOICE_ARBITER \
       USB_CAMERA0_DEVICE USB_CAMERA1_DEVICE USB_CAMERA_FILTER USB_CAMERA_FORMAT \
       USB_CAMERA_MODES USB_CAMERA_MODE
# The ONLY vision program this project can start. There is no fallback binary.
VISION_BIN="${HERE}/vision-local/build-devkit/drone-seminar-usb-vision"
VISION_READY_TIMEOUT_S="${VISION_READY_TIMEOUT_S:-240}"

export TMPDIR="${HERE}/runtime/tmp"
export XDG_CACHE_HOME="${HERE}/runtime/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${HERE}/runtime/pycache"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${HERE}/runtime/tls"

die() { echo "[run.sh] ERROR: $*" >&2; exit 1; }
say() { echo "[run.sh] $*"; }
line() { printf '  %-17s %s\n' "$1" "$2"; }

PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || die "no ${PYTHON} on PATH"
[ -f "${SERVER}" ] || die "missing ${SERVER}"

# The DevKit is the Modalix board (it has the MLA and the USB cameras); everything else
# is "the SDK host". No MIPI camera node is needed to tell the two apart.
if [ "$(uname -m)" = "aarch64" ] && { grep -qi modalix /proc/device-tree/model 2>/dev/null || [ -e /dev/simaai-mem ]; }; then
  ON_BOARD=1
else
  ON_BOARD=0
fi

cfg() {  # cfg <python expression over d> - read config/ui_config.json
  "${PYTHON}" - "${CONFIG}" "$1" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(eval(sys.argv[2]))
PY
}
# Ports and addresses come from config/default.env (UI_PORT, VOICE_PORT, ...), not
# from ui_config.json, so a site change is made in exactly one place.
WEB_PORT="${UI_PORT}"

# ================================================================ UI server
is_our_server() {
  local pid="$1"
  case "${pid}" in ''|*[!0-9]*) return 1 ;; esac
  kill -0 "${pid}" 2>/dev/null || return 1
  tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null | grep -qxF "${SERVER}"
}
our_server_pid() {
  [ -f "${PID_FILE}" ] || return 1
  local pid; pid="$(cat "${PID_FILE}" 2>/dev/null)"
  is_our_server "${pid}" || return 1
  echo "${pid}"
}
our_orphan_pids() {
  local pid
  for pid in $(ls -1 /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ "${pid}" = "$$" ] && continue
    is_our_server "${pid}" && echo "${pid}"
  done
}
stop_pid() {
  local pid="$1" i
  kill -TERM "${pid}" 2>/dev/null
  for i in $(seq 1 20); do kill -0 "${pid}" 2>/dev/null || return 0; sleep 0.25; done
  say "pid ${pid} did not stop on SIGTERM; sending SIGKILL"
  kill -KILL "${pid}" 2>/dev/null
  return 0
}
stop_ui() {
  local stopped=1 pid orphan
  if pid="$(our_server_pid)"; then say "stopping the UI server (pid ${pid})"; stop_pid "${pid}"; stopped=0; fi
  rm -f "${PID_FILE}"
  for orphan in $(our_orphan_pids); do
    say "stopping untracked copy of the UI server (pid ${orphan})"; stop_pid "${orphan}"; stopped=0
  done
  return ${stopped}
}

# ============================================================ vision pipeline
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "${DEVKIT_USER}@${DEVKIT_HOST}")
if [ "${ON_BOARD}" -eq 1 ]; then
  VISION_PID_FILE="${HERE}/runtime/vision-${HOSTTAG}.pid"
  VISION_LOG="${HERE}/runtime/vision-${HOSTTAG}.log"
  SELECTION_FILE="${HERE}/runtime/usb-selection-${HOSTTAG}.env"
  PREFLIGHT_LOG="${HERE}/runtime/preflight-${HOSTTAG}.log"
else
  VISION_PID_FILE="${HERE}/runtime/vision-devkit-${DEVKIT_HOST}.pid"
  VISION_LOG="${HERE}/runtime/vision-devkit-${DEVKIT_HOST}.log"
  SELECTION_FILE="${HERE}/runtime/usb-selection-devkit-${DEVKIT_HOST}.env"
  PREFLIGHT_LOG="${HERE}/runtime/preflight-devkit-${DEVKIT_HOST}.log"
fi
VISION_STARTED_HERE=0

on_devkit() {  # run a shell snippet where the cameras, the MLA and the voice runtime live
  if [ "${ON_BOARD}" -eq 1 ]; then bash -c "$1"; else "${SSH[@]}" "$1"; fi
}
devkit_exec() {  # devkit_exec <program> <args...>   (arguments quoted for the remote shell)
  local q; q="$(printf '%q ' "$@")"
  on_devkit "export TMPDIR='${TMPDIR}' XDG_CACHE_HOME='${XDG_CACHE_HOME}'; cd '${HERE}' && ${q}"
}

# true when <pid> is THIS project's vision pipeline (checked where it runs): either
# the vision binary itself, or - during the first moments of a start - this
# project's scripts/start_pipeline.sh, which then execs the binary in the same pid.
vision_alive() {
  local pid="$1"
  case "${pid}" in ''|*[!0-9]*) return 1 ;; esac
  local check="exe=\$(readlink -f /proc/${pid}/exe 2>/dev/null); [ -n \"\$exe\" ] || exit 1; [ \"\$exe\" = \"\$(readlink -f '${VISION_BIN}')\" ] && exit 0; grep -qaF '${HERE}/scripts/start_pipeline.sh' /proc/${pid}/cmdline 2>/dev/null"
  on_devkit "${check}" 2>/dev/null
}
vision_pid() {
  [ -f "${VISION_PID_FILE}" ] || return 1
  local pid; pid="$(cat "${VISION_PID_FILE}" 2>/dev/null)"
  vision_alive "${pid}" || return 1
  echo "${pid}"
}

start_vision() {
  local pid
  : > "${VISION_LOG}"
  if [ "${ON_BOARD}" -eq 1 ]; then
    [ -x "${VISION_BIN}" ] || die "missing ${VISION_BIN} - build it in the SDK container:
       vision-local/build.sh"
    # setsid without job control does not fork, so $! IS the vision binary after
    # start_pipeline.sh execs it.
    setsid "${HERE}/scripts/start_pipeline.sh" --no-preflight --selection "${SELECTION_FILE}" \
      >>"${VISION_LOG}" 2>&1 < /dev/null &
    pid=$!
  else
    pid="$("${SSH[@]}" "cd '${HERE}' && { setsid scripts/start_pipeline.sh --no-preflight --selection '${SELECTION_FILE}' >>'${VISION_LOG}' 2>&1 < /dev/null & echo \$!; }")"
  fi
  echo "${pid}" > "${VISION_PID_FILE}"
  VISION_STARTED_HERE=1
  say "vision pipeline starting (pid ${pid}$([ "${ON_BOARD}" -eq 0 ] && echo " on ${DEVKIT_HOST}"), log ${VISION_LOG})"
}

stop_vision() {
  local pid
  pid="$(cat "${VISION_PID_FILE}" 2>/dev/null)"
  if ! vision_alive "${pid}"; then rm -f "${VISION_PID_FILE}" "${SELECTION_FILE}"; return 1; fi
  say "stopping the vision pipeline (pid ${pid}, SIGINT - cameras and MLA released cleanly)"
  local stop="kill -INT ${pid}; for i in \$(seq 1 40); do kill -0 ${pid} 2>/dev/null || exit 0; sleep 1; done; kill -TERM ${pid}; sleep 3; kill -0 ${pid} 2>/dev/null && kill -KILL ${pid}; exit 0"
  on_devkit "${stop}"
  # libsima_neat writes /tmp/sima_gst_plugin_scanner_<pid> (a fixed path, TMPDIR is not
  # used) and never removes it; remove the one belonging to the process just stopped.
  on_devkit "rm -f '/tmp/sima_gst_plugin_scanner_${pid}'"
  rm -f "${VISION_PID_FILE}" "${SELECTION_FILE}"
  return 0
}

wait_vision_ready() {
  local t0 now pid
  pid="$(cat "${VISION_PID_FILE}" 2>/dev/null)"
  t0=$(date +%s)
  say "waiting for both USB cameras and both models (typically 30-90 s)"
  while :; do
    if grep -a -q '\[ch0 detection\] \[stats\]' "${VISION_LOG}" 2>/dev/null &&
       grep -a -q '\[ch1 segmentation\] \[stats\]' "${VISION_LOG}" 2>/dev/null; then
      return 0
    fi
    if grep -a -q -E '\[FAIL\]|\[ERR\]|\] ERROR: |setup failed|terminate called' "${VISION_LOG}" 2>/dev/null; then
      return 1
    fi
    now=$(date +%s)
    if [ $((now - t0)) -ge "${VISION_READY_TIMEOUT_S}" ]; then return 1; fi
    if [ $((now - t0)) -ge 10 ] && [ $(( (now - t0) % 15 )) -eq 0 ] && ! vision_alive "${pid}"; then
      say "the vision process (pid ${pid}) exited"
      return 1
    fi
    sleep 1
  done
}

camera_line() {  # camera_line <tag>
  local cam stats
  cam="$(grep -a -m1 -o "\[$1\] camera \"[^\"]*\" ([^)]*)" "${VISION_LOG}" | sed 's/^\[[^]]*\] camera //')"
  stats="$(grep -a "\[$1\] \[stats\]" "${VISION_LOG}" | tail -1 | grep -o 'yolo_fps=[0-9.]*')"
  echo "READY  ${cam} ${stats:+(${stats/=/ })}"
}

insight_line() {
  local host="$1"
  "${PYTHON}" - "${host}" <<'PY' 2>/dev/null || echo "UNREACHABLE (https://${host}:9900)"
import json, ssl, sys, urllib.request
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
host = sys.argv[1]
d = json.load(urllib.request.urlopen("https://%s:9900/api/ingest/stats" % host, timeout=5, context=ctx))
chans = {c["channel"]: c for c in d.get("channels", [])}
video = [ch for ch in (0, 1) if ch in chans and chans[ch].get("active")]
print("CONNECTED  https://%s:9900  video arriving on channel(s) %s" % (host, video or "none yet"))
PY
}

# ============================================================ voice runtime
# voice/board_voice_server.py: Whisper-medium + Qwen3-0.6B on the MLA, ON THE DEVKIT.
VOICE_SERVER="${HERE}/voice/board_voice_server.py"
VOICE_READY_TIMEOUT_S="${VOICE_READY_TIMEOUT_S:-300}"
if [ "${ON_BOARD}" -eq 1 ]; then
  VOICE_PID_FILE="${HERE}/runtime/voice-${HOSTTAG}.pid"
  VOICE_LOG="${HERE}/runtime/voice-${HOSTTAG}.log"
else
  VOICE_PID_FILE="${HERE}/runtime/voice-devkit-${DEVKIT_HOST}.pid"
  VOICE_LOG="${HERE}/runtime/voice-devkit-${DEVKIT_HOST}.log"
fi
VOICE_STARTED_HERE=0

voice_alive() {
  local pid="$1"
  case "${pid}" in ''|*[!0-9]*) return 1 ;; esac
  on_devkit "grep -qaF '${VOICE_SERVER}' /proc/${pid}/cmdline 2>/dev/null" 2>/dev/null
}
voice_pid() {
  [ -f "${VOICE_PID_FILE}" ] || return 1
  local pid; pid="$(cat "${VOICE_PID_FILE}" 2>/dev/null)"
  voice_alive "${pid}" || return 1
  echo "${pid}"
}
voice_health() {  # prints the /health JSON (or nothing)
  on_devkit "curl -s --max-time 4 http://127.0.0.1:${VOICE_PORT}/health" 2>/dev/null
}
start_voice() {
  local pid arbiter_arg="--arbiter"
  [ "${VOICE_ARBITER:-1}" = "0" ] && arbiter_arg="--no-arbiter"
  : > "${VOICE_LOG}"
  local cmd="cd '${HERE}' && export MODEL_ROOT='${MODEL_ROOT}' VOICE_PORT='${VOICE_PORT}' ARBITER_PORT='${ARBITER_PORT}' PYTHONDONTWRITEBYTECODE=1 TMPDIR='${TMPDIR}' XDG_CACHE_HOME='${XDG_CACHE_HOME}' && { setsid '${PYNEAT_ENV}/bin/python3' -u '${VOICE_SERVER}' --port '${VOICE_PORT}' ${arbiter_arg} --arbiter-port '${ARBITER_PORT}' >> '${VOICE_LOG}' 2>&1 < /dev/null & echo \$!; }"
  pid="$(on_devkit "${cmd}")"
  case "${pid}" in ''|*[!0-9]*) die "could not start the voice runtime (see ${VOICE_LOG})" ;; esac
  echo "${pid}" > "${VOICE_PID_FILE}"
  VOICE_STARTED_HERE=1
  say "voice runtime starting on the DevKit (pid ${pid}, Whisper-medium + Qwen3-0.6B on the MLA, log ${VOICE_LOG})"
}
wait_voice_ready() {
  local t0 health pid
  pid="$(cat "${VOICE_PID_FILE}" 2>/dev/null)"
  t0=$(date +%s)
  say "waiting for Whisper-medium and Qwen3-0.6B to load and warm up on the MLA"
  while :; do
    health="$(voice_health)"
    case "${health}" in
      *'"ready": true'*) return 0 ;;
      *'"state": "error"'*) return 1 ;;
    esac
    if [ $(( $(date +%s) - t0 )) -ge "${VOICE_READY_TIMEOUT_S}" ]; then return 1; fi
    if [ $(( $(date +%s) - t0 )) -ge 5 ] && ! voice_alive "${pid}"; then return 1; fi
    sleep 2
  done
}
stop_voice() {
  local pid
  pid="$(cat "${VOICE_PID_FILE}" 2>/dev/null)"
  if ! voice_alive "${pid}"; then rm -f "${VOICE_PID_FILE}"; return 1; fi
  say "stopping the voice runtime (pid ${pid})"
  on_devkit "kill -INT ${pid}; for i in \$(seq 1 20); do kill -0 ${pid} 2>/dev/null || exit 0; sleep 0.5; done; kill -TERM ${pid}; sleep 2; kill -0 ${pid} 2>/dev/null && kill -KILL ${pid}; exit 0"
  rm -f "${VOICE_PID_FILE}"
  return 0
}

voice_line() {
  "${PYTHON}" - "$(voice_health)" <<'PY' 2>/dev/null || echo "unavailable"
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print("unavailable (voice runtime not answering)")
    sys.exit(0)
m = d.get("models", {})
state = "READY" if d.get("ready") else ("loading (%s)" % d.get("state"))
print("%s  on-device MLA: %s + %s" % (state, (m.get("whisper") or {}).get("model_id"),
                                    (m.get("qwen") or {}).get("model_id")))
PY
}

mla_summary() {  # the four AI workloads and the evidence that each runs on the MLA
  local det seg
  echo "  AI workloads on the Modalix MLA:"
  if [ "${START_VISION}" -eq 1 ]; then
    det="$(grep -a '\[ch0 detection\] \[stats\]' "${VISION_LOG}" 2>/dev/null | tail -1 | grep -o 'yolo_fps=[0-9.]*')"
    seg="$(grep -a '\[ch1 segmentation\] \[stats\]' "${VISION_LOG}" 2>/dev/null | tail -1 | grep -o 'yolo_fps=[0-9.]*')"
    printf '    %-22s %-44s %s\n' "YOLO26m Detection" "NEAT C++ Model::Runner  -> Modalix MLA" "RUNNING (${det:-no stats yet})"
    printf '    %-22s %-44s %s\n' "YOLO26m Segmentation" "NEAT C++ Model::Runner  -> Modalix MLA" "RUNNING (${seg:-no stats yet})"
  else
    echo "    YOLO26m Detection / YOLO26m Segmentation: not started (--no-vision)"
  fi
  "${PYTHON}" - "$(voice_health)" <<'PY' 2>/dev/null || echo "    Whisper-medium / Qwen3-0.6B: voice runtime not answering"
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print("    Whisper-medium / Qwen3-0.6B: voice runtime not started")
    sys.exit(0)
models = d.get("models", {})
for key, runtime in (("whisper", "pyneat genai.ASRModel"), ("qwen", "pyneat genai.GenAIModel")):
    m = models.get(key) or {}
    state = "LOADED + WARMED UP" if (m.get("loaded") and d.get("ready")) else "NOT READY"
    print("    %-22s %-44s %s" % (m.get("name", key), "%s  -> %s" % (runtime, m.get("device", "?")), state))
PY
  echo "    (the deterministic command parser is ordinary software logic, not an AI workload)"
}

ui_url() {
  local ip
  if [ "${ON_BOARD}" -eq 1 ]; then
    ip="$(ip -4 -o addr show 2>/dev/null | awk '$2 != "lo" {sub(/\/.*/, "", $4); print $4; exit}')"
  else
    ip="127.0.0.1"
  fi
  echo "https://${ip:-127.0.0.1}:${WEB_PORT}"
}

insight_host() {
  local h; h="$(cfg 'd["insight"].get("host") or ""')"
  case "${h}" in ''|local|localhost|127.0.0.1)
    if [ "${ON_BOARD}" -eq 1 ]; then h="${CONTAINER_HOST_IP:-10.42.0.1}"; else h="127.0.0.1"; fi ;;
  esac
  echo "${h}"
}

# ================================================================ preflight
PF_FAIL=0
pf_ok()   { printf '  OK    %s\n' "$*"; }
pf_warn() { printf '  WARN  %s\n' "$*"; }
pf_fail() { printf '  FAIL  %s\n' "$*"; PF_FAIL=$((PF_FAIL + 1)); }

# Processes of THIS project copy on the DevKit that no pid file accounts for. The
# patterns are regex-escaped, so the scanning shell never matches its own command line.
stale_scan() {
  local voice_re="${VOICE_SERVER//./\\.}"
  on_devkit "vb=\$(readlink -f '${VISION_BIN}' 2>/dev/null); for p in /proc/[0-9]*; do pid=\${p#/proc/}; [ \"\$pid\" = \"\$\$\" ] && continue; exe=\$(readlink -f \$p/exe 2>/dev/null); [ -n \"\$vb\" ] && [ \"\$exe\" = \"\$vb\" ] && echo \"vision pid \$pid\"; tr '\\0' ' ' < \$p/cmdline 2>/dev/null | grep -q '${voice_re}' && echo \"voice pid \$pid\"; done; exit 0" 2>/dev/null
}

port_busy_here() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
port_busy_devkit() { on_devkit "(exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null" 2>/dev/null; }

camera_probe() {
  local args=(--config "${HERE}/config/vision.yaml" --probe-cameras --selection-out "${SELECTION_FILE}"
              --usb-format "${USB_CAMERA_FORMAT}" --usb-modes "${USB_CAMERA_MODES}")
  [ -n "${USB_CAMERA0_DEVICE}" ] && args+=(--camera0 "${USB_CAMERA0_DEVICE}")
  [ -n "${USB_CAMERA1_DEVICE}" ] && args+=(--camera1 "${USB_CAMERA1_DEVICE}")
  [ -n "${USB_CAMERA_FILTER}" ] && args+=(--camera-filter "${USB_CAMERA_FILTER}")
  [ -n "${USB_CAMERA_MODE}" ] && args+=(--usb-mode "${USB_CAMERA_MODE}")
  rm -f "${SELECTION_FILE}"
  devkit_exec "${VISION_BIN}" "${args[@]}" > "${PREFLIGHT_LOG}" 2>&1
}

print_camera_selection() {
  # shellcheck disable=SC1090
  . "${SELECTION_FILE}"
  local mode="${USB_SELECTED_MODE}" fmt res fps i dev card path ident pfps
  fmt="${mode%%:*}"; res="${mode#*:}"; fps="${res#*@}"; res="${res%@*}"
  for i in 0 1; do
    eval "dev=\${USB_SELECTED_CAMERA${i}_DEVICE:-}; card=\${USB_SELECTED_CAMERA${i}_CARD:-}; path=\${USB_SELECTED_CAMERA${i}_USB_PATH:-}; ident=\${USB_SELECTED_CAMERA${i}_IDENTITY:-}; pfps=\${USB_SELECTED_CAMERA${i}_PROBE_FPS:-}"
    if [ "${i}" -eq 0 ]; then echo "  USB Camera 0  (LEFT,  ch0 -> YOLO26m detection)"; else echo "  USB Camera 1  (RIGHT, ch1 -> YOLO26m-seg segmentation)"; fi
    line "  device:" "${dev}   \"${card}\"  usb ${path}"
    line "  identity:" "${ident}"
    line "  format:" "${fmt}$([ "${fmt}" = "MJPG" ] && echo " (hardware JPEG decode -> NV12)")"
    line "  resolution:" "${res}"
    line "  fps:" "${fps}${pfps:+  (probe measured ${pfps})}"
  done
  if [ "${USB_SELECTED_PREFERRED_MODE:-${mode}}" != "${mode}" ]; then
    echo "  NOTE: preferred ${USB_SELECTED_PREFERRED_MODE} could not be streamed by both cameras at once;"
    echo "        using ${mode} (probe details: ${PREFLIGHT_LOG})"
  fi
}

preflight() {
  local out stale
  say "preflight checks$([ "${ON_BOARD}" -eq 0 ] && echo " (DevKit ${DEVKIT_USER}@${DEVKIT_HOST})")"
  if [ "${ON_BOARD}" -eq 0 ]; then
    "${SSH[@]}" true 2>/dev/null || die "cannot reach the DevKit at ${DEVKIT_USER}@${DEVKIT_HOST}"
    pf_ok "DevKit ${DEVKIT_HOST} reachable"
  fi

  # -- no stale instance
  local stale_files=()
  [ -f "${VISION_PID_FILE}" ] && { if vision_pid >/dev/null; then pf_fail "vision pipeline of this project already running (pid $(cat "${VISION_PID_FILE}"))"; else stale_files+=("${VISION_PID_FILE}"); fi; }
  [ -f "${VOICE_PID_FILE}" ] && { if voice_pid >/dev/null; then pf_fail "voice runtime of this project already running (pid $(cat "${VOICE_PID_FILE}"))"; else stale_files+=("${VOICE_PID_FILE}"); fi; }
  if [ ${#stale_files[@]} -gt 0 ]; then
    rm -f "${stale_files[@]}"
    pf_warn "removed stale pid file(s) of processes that no longer run: ${stale_files[*]##*/}"
  fi
  stale="$(stale_scan)"
  if [ -n "${stale}" ]; then
    pf_fail "untracked process(es) of this project are running: $(echo "${stale}" | tr '\n' ';')  ->  ./run.sh --stop-all"
  else
    pf_ok "no other instance of this project is running"
  fi

  # -- binaries, libraries, GStreamer elements
  if [ "${START_VISION}" -eq 1 ]; then
    if on_devkit "[ -x '${VISION_BIN}' ]"; then
      out="$(on_devkit "ldd '${VISION_BIN}' 2>&1 | grep 'not found'")"
      if [ -z "${out}" ]; then pf_ok "vision binary and its libraries (${VISION_BIN#${HERE}/})"; else pf_fail "vision binary has unresolved libraries: $(echo "${out}" | tr -s ' ' | tr '\n' ';')"; fi
    else
      pf_fail "missing ${VISION_BIN} - build it in the SDK container: vision-local/build.sh"
    fi
    out="$(on_devkit "export XDG_CACHE_HOME='${XDG_CACHE_HOME}'; for e in v4l2src neatdecoder neatencoder h264parse rtph264pay udpsink appsink; do gst-inspect-1.0 \$e >/dev/null 2>&1 || echo \$e; done")"
    if [ -z "${out}" ]; then pf_ok "GStreamer elements v4l2src neatdecoder neatencoder h264parse rtph264pay udpsink appsink"; else pf_fail "GStreamer element(s) missing: $(echo "${out}" | tr '\n' ' ')"; fi
  fi

  # -- models
  local models_rc=0 wanted="yolo26m,yolo26m_seg,whisper_medium,qwen3_0_6b"
  [ "${START_VISION}" -eq 0 ] && wanted="whisper_medium,qwen3_0_6b"
  [ "${START_VOICE}" = "no" ] && wanted="yolo26m,yolo26m_seg"
  out="$(devkit_exec env MODEL_ROOT="${MODEL_ROOT}" python3 scripts/check_models.py --only "${wanted}" 2>&1)" || models_rc=1
  while IFS= read -r l; do
    [ -z "${l}" ] && continue
    case "${l}" in PASS*) pf_ok "model ${l#PASS  }" ;; *) pf_fail "model ${l#FAIL  }" ;; esac
  done <<< "${out}"
  [ "${models_rc}" -ne 0 ] && [ -z "${out}" ] && pf_fail "scripts/check_models.py failed on the DevKit"

  # -- voice runtime
  if [ "${START_VOICE}" != "no" ]; then
    if on_devkit "'${PYNEAT_ENV}/bin/python3' -c 'import pyneat; g = pyneat.genai; assert hasattr(g, \"ASRModel\") and hasattr(g, \"GenAIModel\")'" >/dev/null 2>&1; then
      pf_ok "pyneat with genai.ASRModel + genai.GenAIModel (${PYNEAT_ENV})"
    else
      pf_fail "pyneat genai runtime not importable from ${PYNEAT_ENV}/bin/python3 (set PYNEAT_ENV)"
    fi
  fi

  # -- ports
  if port_busy_here "${UI_PORT}"; then pf_fail "tcp ${UI_PORT} (UI) already in use on this machine"; else pf_ok "tcp ${UI_PORT} (UI) free"; fi
  if [ "${START_VOICE}" != "no" ]; then
    if port_busy_devkit "${VOICE_PORT}"; then pf_fail "tcp ${VOICE_PORT} (voice) already in use on the DevKit"; else pf_ok "tcp ${VOICE_PORT} (voice) free"; fi
  fi
  if [ "${START_VISION}" -eq 1 ]; then
    if port_busy_devkit "${ARBITER_PORT}"; then pf_fail "tcp ${ARBITER_PORT} (MLA arbiter) already in use on the DevKit"; else pf_ok "tcp ${ARBITER_PORT} (MLA arbiter) free"; fi
  fi

  # -- USB cameras: discovery, identity, capture nodes, a mode both can stream at once
  if [ "${START_VISION}" -eq 1 ] && [ "${PF_FAIL}" -eq 0 ]; then
    say "probing the USB cameras (every candidate mode is streamed on both cameras at once)"
    if camera_probe && [ -f "${SELECTION_FILE}" ]; then
      grep -E '^discovered |^\s+camera[01] \(|^capture mode probe|^\s+mode ' "${PREFLIGHT_LOG}" | sed 's/^/  /'
      pf_ok "two USB UVC capture cameras selected and a common capture mode streams on both"
      print_camera_selection
      local i dev holders
      for i in 0 1; do
        eval "dev=\${USB_SELECTED_CAMERA${i}_DEVICE:-}"
        holders="$(on_devkit "for p in /proc/[0-9]*/fd/*; do [ \"\$(readlink \$p 2>/dev/null)\" = '${dev}' ] && echo \$p | cut -d/ -f3; done | sort -u | tr '\n' ' '")"
        if [ -n "${holders}" ]; then pf_fail "${dev} is held open by pid(s) ${holders}"; else pf_ok "${dev} not held open by any process"; fi
      done
    else
      sed 's/^/        /' "${PREFLIGHT_LOG}"
      pf_fail "USB camera preflight failed (full output: ${PREFLIGHT_LOG})"
    fi
  fi

  if [ "${PF_FAIL}" -ne 0 ]; then
    rm -f "${SELECTION_FILE}"
    die "preflight failed with ${PF_FAIL} problem(s); nothing was started"
  fi
  say "preflight passed"
}


# ================================================================ arguments
ACTION=run
BACKGROUND=0
START_VOICE=auto
START_VISION=1
ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --stop)         ACTION=stop ;;
    --stop-all)     ACTION=stopall ;;
    --status)       ACTION=status ;;
    --check)        ACTION=check ;;
    --selftest)     ACTION=selftest ;;
    --background)   BACKGROUND=1 ;;
    --voice)        START_VOICE=yes ;;
    --no-voice)     START_VOICE=no ;;
    --no-vision)    START_VISION=0 ;;
    --help|-h)      sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              ARGS+=("${arg}") ;;
  esac
done

case "${ACTION}" in
  status)
    if pid="$(our_server_pid)"; then line "Demo UI:" "running, pid ${pid}, $(ui_url)"; else line "Demo UI:" "not running"; fi
    if pid="$(vision_pid)"; then
      line "Vision pipeline:" "running, pid ${pid}$([ "${ON_BOARD}" -eq 0 ] && echo " on ${DEVKIT_HOST}")"
      line "Camera 0:" "$(camera_line 'ch0 detection')"
      line "Camera 1:" "$(camera_line 'ch1 segmentation')"
    else
      line "Vision pipeline:" "not running"
    fi
    line "Insight:" "$(insight_line "$(insight_host)")"
    if pid="$(voice_pid)"; then line "Voice backend:" "$(voice_line)  (pid ${pid})"; else line "Voice backend:" "not running"; fi
    exit 0
    ;;
  stop)
    stop_ui || say "no UI server of ours was running"
    say "(the vision pipeline and voice server were left running; ./run.sh --stop-all stops them)"
    exit 0
    ;;
  stopall)
    stop_ui || say "no UI server of ours was running"
    stop_voice || say "no voice runtime of ours was running"
    stop_vision || say "no vision pipeline of ours was running"
    say "done. Insight and processes of other projects were not touched."
    exit 0
    ;;
  check)
    if [ "${ON_BOARD}" -eq 1 ]; then exec "${HERE}/scripts/check_environment.sh"; fi
    exec "${SSH[@]}" "'${HERE}/scripts/check_environment.sh'"
    ;;
  selftest)
    exec "${PYTHON}" -u "${HERE}/scripts/selftest.py" ${ARGS[@]+"${ARGS[@]}"}
    ;;
esac

if pid="$(our_server_pid)"; then
  die "the UI server is already running as pid ${pid}
       ./run.sh --stop-all   stops the whole demo"
fi
for orphan in $(our_orphan_pids); do
  die "an untracked copy of the UI server is running as pid ${orphan}
       ./run.sh --stop-all   stops it"
done

SERVER_PID=""
SHUTDOWN_DONE=0
shutdown_all() {
  [ "${SHUTDOWN_DONE}" -eq 1 ] && return 0
  SHUTDOWN_DONE=1
  trap '' INT TERM EXIT
  echo ""
  say "shutting down"
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    say "stopping the UI server (pid ${SERVER_PID})"; stop_pid "${SERVER_PID}"
  fi
  rm -f "${PID_FILE}"
  if [ "${VOICE_STARTED_HERE}" -eq 1 ]; then stop_voice || true; fi
  if [ "${VISION_STARTED_HERE}" -eq 1 ]; then stop_vision || true; fi
  rm -f "${SELECTION_FILE}"
  say "done - everything this run started is stopped (Insight untouched)."
}
[ "${BACKGROUND}" -eq 0 ] && { trap shutdown_all INT TERM; trap shutdown_all EXIT; }

echo "=================================================================="
echo " Palette NEAT SDK Demo (dual USB cameras) - $([ "${ON_BOARD}" -eq 1 ] && echo "on the DevKit" || echo "on the SDK host (vision on ${DEVKIT_HOST})")"
echo "=================================================================="

# ---- 0. preflight -------------------------------------------------------------
preflight

# ---- 1. vision --------------------------------------------------------------
if [ "${START_VISION}" -eq 1 ]; then
  start_vision
  if ! wait_vision_ready; then
    say "VISION PIPELINE FAILED TO BECOME READY. Last lines of ${VISION_LOG}:"
    grep -a -v -E ' INFO | WARN ' "${VISION_LOG}" | tail -n 25 >&2
    [ "${VISION_STARTED_HERE}" -eq 1 ] && stop_vision
    VISION_STARTED_HERE=0
    exit 1
  fi
fi

# ---- 2. voice (on the DevKit MLA; never on the SDK host) ----------------------
if [ "${START_VOICE}" != "no" ]; then
  start_voice
  if ! wait_voice_ready; then
    say "VOICE RUNTIME FAILED TO BECOME READY. Last lines of ${VOICE_LOG}:"
    tail -n 25 "${VOICE_LOG}" >&2
    [ "${VOICE_STARTED_HERE}" -eq 1 ] && stop_voice
    VOICE_STARTED_HERE=0
    [ "${VISION_STARTED_HERE}" -eq 1 ] && stop_vision
    VISION_STARTED_HERE=0
    exit 1
  fi
fi

# ---- 3. UI --------------------------------------------------------------------
SERVER_ARGS=(--bind "${UI_BIND}" --port "${UI_PORT}" --sync-buffer-ms "${SYNC_BUFFER_MS}"
             --insight-host "${INSIGHT_HOST}" --voice-host 127.0.0.1 --voice-port "${VOICE_PORT}"
             --capture-selection "${SELECTION_FILE}")
"${PYTHON}" -u "${SERVER}" "${SERVER_ARGS[@]}" ${ARGS[@]+"${ARGS[@]}"} >>"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
for i in $(seq 1 30); do
  curl -sk --max-time 1 -o /dev/null "https://127.0.0.1:${WEB_PORT}/" && break
  curl -s --max-time 1 -o /dev/null "http://127.0.0.1:${WEB_PORT}/" && break
  kill -0 "${SERVER_PID}" 2>/dev/null || break
  sleep 0.5
done
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  say "the UI server exited immediately; last lines of ${LOG_FILE}:"
  tail -n 30 "${LOG_FILE}" >&2
  exit 1
fi

# ---- 4. summary ---------------------------------------------------------------
echo ""
if [ "${START_VISION}" -eq 1 ]; then
  line "Vision pipeline:" "READY  (pid $(cat "${VISION_PID_FILE}"), log ${VISION_LOG})"
  line "Camera 0 (LEFT):" "$(camera_line 'ch0 detection')  -> YOLO26m detection"
  line "Camera 1 (RIGHT):" "$(camera_line 'ch1 segmentation')  -> YOLO26m-seg segmentation"
else
  line "Vision pipeline:" "not started (--no-vision)"
fi
line "Insight:" "$(insight_line "$(insight_host)")"
line "Voice backend:" "$(voice_line)"
line "Demo UI:" "$(ui_url)"
echo ""
mla_summary
echo ""
if [ "${BACKGROUND}" -eq 1 ]; then
  say "running in the background.  ./run.sh --stop-all   stops the whole demo"
  exit 0
fi
say "Ctrl+C stops everything this run started (UI log: ${LOG_FILE})"

wait "${SERVER_PID}"
STATUS=$?
SERVER_PID=""
shutdown_all
exit "${STATUS}"
