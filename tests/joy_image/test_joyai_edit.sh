#!/usr/bin/env bash
# Smoke-test the JoyAI-Image-Edit pipeline end to end (image edit).
#
# Env overrides:
#   MODEL     served model id (default raw JoyAI dir with trailing '/')
#   PORT      server port (default 8094)
#   API_URL   endpoint (default http://127.0.0.1:${PORT}/v1/chat/completions)
#   IMAGE     input image (default auto-generated red-circle PNG)
#   OUT       output edited image (default ./joyai_edit_test.png)
#   PROMPT    edit instruction (default "Turn the red circle blue")
#   HEIGHT / WIDTH / NUM_INFERENCE_STEPS / CFG_SCALE / SEED
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-joyai-edit}"
PORT="${PORT:-8094}"
API_URL="${API_URL:-http://127.0.0.1:${PORT}/v1/chat/completions}"
IMAGE="${IMAGE:-/tmp/joyai_test_input.png}"
OUT="${OUT:-/tmp/joyai_edit_test.png}"
PROMPT="${PROMPT:-Turn the red circle blue}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${NUM_INFERENCE_STEPS:-30}"
CFG="${CFG_SCALE:-4.0}"
SEED="${SEED:-0}"

python3 "${HERE}/joyai_edit_client.py" \
    --api-url "${API_URL}" \
    --model "${MODEL}" \
    --image "${IMAGE}" \
    --prompt "${PROMPT}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --steps "${STEPS}" \
    --cfg-scale "${CFG}" \
    --seed "${SEED}" \
    --output "${OUT}" \
    --verify

echo "JoyAI-Image-Edit test OK: ${OUT}"
