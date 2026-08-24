#!/usr/bin/env bash
# Send a Wan2.2-T2V smoke-test request to the vllm-omni server.
#
# Env overrides:
#   API_URL  endpoint (default http://127.0.0.1:8091/v1/videos/sync)
#   OUT      output mp4 (default ./wan22_t2v_test.mp4)
#   PROMPT   text prompt
#   HEIGHT / WIDTH / NUM_FRAMES / NUM_INFERENCE_STEPS
#   FLOW_SHIFT / GUIDANCE / FPS / SEED
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="${API_URL:-http://127.0.0.1:8091/v1/videos/sync}"
OUT="${OUT:-${HERE}/wan22_t2v_test.mp4}"
PROMPT="${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-49}"
STEPS="${NUM_INFERENCE_STEPS:-20}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
GUIDANCE="${GUIDANCE:-4.0}"
FPS="${FPS:-16}"
SEED="${SEED:-65535}"

curl -sS -X POST "${API_URL}" \
    -F "prompt=${PROMPT}" \
    -F "height=${HEIGHT}" \
    -F "width=${WIDTH}" \
    -F "num_frames=${NUM_FRAMES}" \
    -F "num_inference_steps=${STEPS}" \
    -F "guidance_scale=${GUIDANCE}" \
    -F "boundary_ratio=0.875" \
    -F "flow_shift=${FLOW_SHIFT}" \
    -F "fps=${FPS}" \
    -F "seed=${SEED}" \
    -o "${OUT}" \
    -w 'HTTP %{http_code} time %{time_total}s size %{size_download}\n'

echo "output: ${OUT}"
