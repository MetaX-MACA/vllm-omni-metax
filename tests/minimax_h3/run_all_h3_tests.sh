#!/usr/bin/env bash
# End-to-end MiniMax-H3 smoke tests (T2VA / FL2VA / Ref2VA), all with audio.
#
# Flow: prepare assets -> start FL2VA server -> T2VA + FL2VA -> stop ->
# start Ref2VA server -> image+audio + video reference -> stop -> verify.
#
# Env overrides:
#   MODEL_ROOT       weights root            (default /external/ai/share/lli/MiniMax-H3)
#   H3_OUT_DIR       output mp4s             (default /tmp/h3_tests)
#   H3_LOG_DIR       server logs + pids      (default /tmp/h3_tests_logs)
#   H3_ASSETS_DIR    generated assets        (default tests/minimax_h3/assets)
#   H3_DEVICES       GPUs for both servers   (default 0,1)
#   NUM_INFERENCE_STEPS / H3_DURATION        (default 20 / 4.0)
#   RESTART_FL2VA    1 to leave FL2VA server running at the end
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-/external/ai/share/lli/MiniMax-H3}"
OUT_DIR="${H3_OUT_DIR:-/tmp/h3_tests}"
LOG_DIR="${H3_LOG_DIR:-/tmp/h3_tests_logs}"
ASSETS="${H3_ASSETS_DIR:-${HERE}/assets}"
export H3_ASSETS_DIR="${ASSETS}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
export H3_DURATION="${H3_DURATION:-4.0}"
export H3_DEVICES="${H3_DEVICES:-0,1}"
export MODEL_ROOT

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${ASSETS}"

echo "== sanity checks =="
for p in "${MODEL_ROOT}/FL2VA/model_index.json" "${MODEL_ROOT}/Ref2VA/model_index.json"; do
    [[ -f "${p}" ]] || { echo "error: missing ${p}"; exit 1; }
done
command -v ffmpeg >/dev/null || echo "warning: ffmpeg not on PATH (Ref2VA video reference needs it)"
command -v ffprobe >/dev/null || echo "warning: ffprobe not on PATH (Ref2VA video reference needs it)"
command -v vllm >/dev/null || { echo "error: vllm not on PATH"; exit 1; }

echo "== preparing assets =="
python3 "${HERE}/make_assets.py" "${ASSETS}"

FL2VA_PID=""
REF2VA_PID=""
start_server() { # $1 = script, $2 = port, $3 = pidfile
    local script="$1" port="$2" pidfile="$3"
    H3_PORT="${port}" nohup bash "${script}" > "${LOG_DIR}/$(basename "${script}" .sh).log" 2>&1 &
    echo $! > "${pidfile}"
    echo "server starting on port ${port} (pid $(cat "${pidfile}"))"
}

wait_ready() { # $1 = port, $2 = timeout seconds
    local port="$1" timeout="$2" waited=0
    while [[ "${waited}" -lt "${timeout}" ]]; do
        if curl -s -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            echo "server ready on port ${port} (${waited}s)"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "error: server on port ${port} not ready within ${timeout}s" >&2
    return 1
}

stop_server() { # $1 = pidfile
    local pidfile="$1"
    if [[ -f "${pidfile}" ]]; then
        local pid
        pid="$(cat "${pidfile}")"
        kill "${pid}" 2>/dev/null || true
        sleep 5
        kill -9 "${pid}" 2>/dev/null || true
        rm -f "${pidfile}"
        echo "server stopped (was pid ${pid})"
    fi
}

FAILED=0

echo "== FL2VA tests (T2VA + first-frame) =="
start_server "${HERE}/serve_h3_fl2va.sh" 8091 "${LOG_DIR}/fl2va.pid"
FL2VA_PID="$(cat "${LOG_DIR}/fl2va.pid")"
wait_ready 8091 900

export API_URL="http://127.0.0.1:8091/v1/videos/sync"
export OUT="${OUT_DIR}/t2va.mp4"
bash "${HERE}/test_h3_t2va.sh" || FAILED=1
export OUT="${OUT_DIR}/fl2va_first_frame.mp4"
bash "${HERE}/test_h3_fl2va.sh" || FAILED=1
stop_server "${LOG_DIR}/fl2va.pid"

echo "== Ref2VA tests (image+audio, video reference) =="
start_server "${HERE}/serve_h3_ref2va.sh" 8092 "${LOG_DIR}/ref2va.pid"
REF2VA_PID="$(cat "${LOG_DIR}/ref2va.pid")"
wait_ready 8092 900

export API_URL="http://127.0.0.1:8092/v1/videos/sync"
export OUT_DIR="${OUT_DIR}"
export VIDEO_REF="${OUT_DIR}/fl2va_first_frame.mp4"
bash "${HERE}/test_h3_ref2va.sh" || FAILED=1
stop_server "${LOG_DIR}/ref2va.pid"

if [[ "${RESTART_FL2VA:-0}" == "1" ]]; then
    echo "== restarting FL2VA server (RESTART_FL2VA=1) =="
    start_server "${HERE}/serve_h3_fl2va.sh" 8091 "${LOG_DIR}/fl2va.pid"
fi

echo "== verifying outputs =="
python3 "${HERE}/verify_mp4.py" \
    "${OUT_DIR}/t2va.mp4" \
    "${OUT_DIR}/fl2va_first_frame.mp4" \
    "${OUT_DIR}/ref2va_image_audio.mp4" \
    "${OUT_DIR}/ref2va_video_ref.mp4" \
    || FAILED=1

if [[ "${FAILED}" == "0" ]]; then
    echo "ALL H3 TESTS PASSED"
else
    echo "SOME H3 TESTS FAILED (see logs in ${LOG_DIR})"
    exit 1
fi
