#!/usr/bin/env bash
# MiniMax-H3 web client (FastAPI page proxying to the local vllm-omni server).
#
# Env overrides:
#   VLLM_URL            backend vllm-omni endpoint (default http://127.0.0.1:8091)
#   H3_WEB_PORT         client port                 (default 8090)
#   H3_WEB_OUTPUT_DIR   generated mp4 dir           (default /tmp/h3_web_outputs)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:8091}"
export H3_WEB_OUTPUT_DIR="${H3_WEB_OUTPUT_DIR:-/tmp/h3_web_outputs}"
PORT="${H3_WEB_PORT:-8090}"

echo "MiniMax-H3 web client starting on port ${PORT} (backend: ${VLLM_URL})"
exec python3 "${HERE}/web_client.py" --host 0.0.0.0 --port "${PORT}"
