#!/usr/bin/env bash
# Start JoyAI-Image-Und (Qwen3-VL text encoder) with vllm-omni --omni.
#
# The metax plugin registers a single-LLM-stage qwen3_vl omni pipeline so the
# raw checkpoint serves directly (no conversion / mirror files).
#
# Env overrides:
#   MODEL   JoyAI-Image-Und dir (default below)
#   PORT    server port (default 8095)
#   TP      tensor parallel size (default 2)
#   GPU_MEM gpu-memory-utilization (default 0.9)
#   CUDA_VISIBLE_DEVICES
set -euo pipefail

MODEL="${MODEL:-/mxstorage/pde_ai/models/llm/Diffusion-models/JoyAI-Image-Edit/JoyAI-Image-Und}"
PORT="${PORT:-8095}"
TP="${TP:-2}"
GPU_MEM="${GPU_MEM:-0.9}"

if [[ ! -f "${MODEL}/config.json" ]]; then
    echo "error: JoyAI-Image-Und checkpoint not found at ${MODEL}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "JoyAI-Image-Und server: model=${MODEL} port=${PORT} tp=${TP} gpu_mem=${GPU_MEM}"
exec vllm serve "${MODEL}" \
    --omni \
    --served-model-name joyai-und \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEM}"
