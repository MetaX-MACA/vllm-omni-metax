#!/usr/bin/env bash
# FL2VA first-frame: first frame image -> video + synchronized audio
# (FL2VA partition).
#
# Env overrides:
#   API_URL         server endpoint (default http://127.0.0.1:8091/v1/videos/sync)
#   OUT             output mp4 path  (default /tmp/h3_tests/fl2va_first_frame.mp4)
#   H3_ASSETS_DIR   dir with first_frame.png (default tests/minimax_h3/assets)
#   NUM_INFERENCE_STEPS / H3_DURATION  see test_h3_t2va.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="${API_URL:-http://127.0.0.1:8091/v1/videos/sync}"
OUT="${OUT:-/tmp/h3_tests/fl2va_first_frame.mp4}"
ASSETS="${H3_ASSETS_DIR:-${HERE}/assets}"
STEPS="${NUM_INFERENCE_STEPS:-20}"
DURATION="${H3_DURATION:-4.0}"
FIRST_FRAME="${ASSETS}/first_frame.png"

if [[ ! -f "${FIRST_FRAME}" ]]; then
    echo "error: ${FIRST_FRAME} not found; run make_assets.py first" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"

curl -sS -X POST "${API_URL}" \
    -F 'prompt=A small red fox stands in a snowy blue-purple forest at dawn. It turns its head and blinks, snow crunches softly under its paws, wind whispers through the pines, and a distant bird chirps.' \
    -F fps=24 \
    -F "num_inference_steps=${STEPS}" \
    -F flow_shift=12 \
    -F seed=2101 \
    -F "extra_params={\"task\":\"fl2va\",\"duration\":${DURATION},\"audio_flow_shift\":3.0}" \
    -F "input_reference=@${FIRST_FRAME};type=image/png" \
    -o "${OUT}" \
    -w 'HTTP %{http_code} time %{time_total}s size %{size_download}\n'

echo "FL2VA first-frame output: ${OUT}"
