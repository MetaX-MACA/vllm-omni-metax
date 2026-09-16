#!/usr/bin/env bash
# Start the JoyAI-Image-Edit pipeline (vllm-omni) on MetaX.
#
# The raw MetaX model-registry checkpoint is served directly; the metax plugin
# builds the Diffusers mirror on first use (see
# src/vllm_omni_metax/patches/joy_image_model_builder.py).
#
# Env overrides:
#   MODEL        raw JoyAI-Image-Edit checkpoint dir (default below)
#   PORT         server port (default 8094)
#   MIRROR_DIR   VLLM_OMNI_METAX_JOYAI_MIRROR_DIR (default
#                /sw_home/lli/models/JoyAI-Image-Edit-Diffusers)
#   TENSOR_PARALLEL   DiT tensor parallel size (default 1; >1 unsupported)
#   VAE_USE_TILING    1 to enable VAE tiling/slicing (default 1; needed to
#                     avoid decode OOM on 64 GB MXC500)
#   CUDA_VISIBLE_DEVICES
set -euo pipefail

MODEL="${MODEL:-/mxstorage/pde_ai/models/llm/Diffusion-models/JoyAI-Image-Edit}"
PORT="${PORT:-8094}"
MIRROR_DIR="${MIRROR_DIR:-/sw_home/lli/models/JoyAI-Image-Edit-Diffusers}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
VAE_USE_TILING="${VAE_USE_TILING:-1}"

if [[ ! -d "${MODEL}/transformer" || ! -f "${MODEL}/transformer/transformer.pth" ]]; then
    echo "error: raw JoyAI-Image-Edit checkpoint not found at ${MODEL}" >&2
    exit 1
fi

ARGS=(
    --omni
    --served-model-name joyai-edit
    --host 0.0.0.0
    --port "${PORT}"
    --trust-remote-code
    --tensor-parallel-size "${TENSOR_PARALLEL}"
    --enforce-eager
    --init-timeout 1200
    --stage-init-timeout 900
)
if [[ "${VAE_USE_TILING}" == "1" ]]; then
    ARGS+=(--vae-use-tiling --vae-use-slicing)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_OMNI_METAX_JOYAI_MIRROR_DIR="${MIRROR_DIR}"

echo "JoyAI-Image-Edit server: model=${MODEL} port=${PORT} mirror=${MIRROR_DIR}"
exec vllm serve "${MODEL}" "${ARGS[@]}"
