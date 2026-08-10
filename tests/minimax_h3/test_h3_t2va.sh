#!/usr/bin/env bash
# T2VA: text -> video + synchronized audio (FL2VA partition).
#
# Env overrides:
#   API_URL                 server endpoint (default http://127.0.0.1:8091/v1/videos/sync)
#   OUT                     output mp4 path  (default /tmp/h3_tests/t2va.mp4)
#   NUM_INFERENCE_STEPS     denoise steps    (default 20; smoke test)
#   H3_DURATION             seconds          (default 4.0)
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8091/v1/videos/sync}"
OUT="${OUT:-/tmp/h3_tests/t2va.mp4}"
STEPS="${NUM_INFERENCE_STEPS:-20}"
DURATION="${H3_DURATION:-4.0}"

mkdir -p "$(dirname "${OUT}")"

curl -sS -X POST "${API_URL}" \
    -F 'prompt=A small red fox walks through a snowy blue-purple forest at dawn; footsteps crunch softly in the snow while birds chirp in the distance.' \
    -F width=1024 \
    -F height=576 \
    -F aspect_ratio=16:9 \
    -F fps=24 \
    -F "num_inference_steps=${STEPS}" \
    -F flow_shift=12 \
    -F seed=1101 \
    -F "extra_params={\"task\":\"t2va\",\"duration\":${DURATION},\"audio_flow_shift\":3.0}" \
    -o "${OUT}" \
    -w 'HTTP %{http_code} time %{time_total}s size %{size_download}\n'

echo "T2VA output: ${OUT}"
