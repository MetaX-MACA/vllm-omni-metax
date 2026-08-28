#!/usr/bin/env bash
# Smoke-test JoyAI-Image-Und (Qwen3-VL) through the vllm-omni --omni server.
#
# Env overrides:
#   MODEL     served model id (default JoyAI-Image-Und dir with trailing '/')
#   PORT      server port (default 8095)
#   API_URL   endpoint (default http://127.0.0.1:${PORT}/v1/chat/completions)
#   IMAGE     image for the vision check (default: generated blue circle)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-joyai-und}"
PORT="${PORT:-8095}"
API_URL="${API_URL:-http://127.0.0.1:${PORT}/v1/chat/completions}"
IMAGE="${IMAGE:-}"

python3 "${HERE}/joy_und_client.py" \
    --api-url "${API_URL}" \
    --model "${MODEL}" \
    ${IMAGE:+--image "${IMAGE}"}

echo "JoyAI-Image-Und test OK"
