#!/usr/bin/env bash
# Start Wan2.2-T2V-A14B-Diffusers via vllm-omni 0.26 on MetaX C500.
#
# Validated profile on 4 x 64 GB C500: TP4 (no HSDP / USP).  The HSDP + USP4
# profile boots too but currently produces banded/color-flickering output on
# the MetaX stack (see README.md); keep HSDP off unless it is fixed upstream.
#
# Env overrides:
#   MODEL   model dir (default below)
#   PORT    server port (default 8091)
#   TENSOR_PARALLEL   DiT tensor parallel size (default 4)
#   ULYSESS_DEGREE    Ulysses sequence parallel degree (default 1)
#   USE_HSDP          1 to enable HSDP weight sharding (default 0)
#   VAE_PATCH_PARALLEL_SIZE   VAE patch parallel size (default 4)
#   VAE_USE_TILING    1 to enable VAE tiling (default 1)
set -euo pipefail

MODEL="${MODEL:-/mxstorage/pde_ai/models/llm/Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8091}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
ULYSESS_DEGREE="${ULYSESS_DEGREE:-1}"
USE_HSDP="${USE_HSDP:-0}"
VAE_PATCH_PARALLEL_SIZE="${VAE_PATCH_PARALLEL_SIZE:-4}"
VAE_USE_TILING="${VAE_USE_TILING:-1}"

ARGS=(
    --omni
    --host 0.0.0.0
    --port "${PORT}"
    --trust-remote-code
    --boundary-ratio 0.875
    --flow-shift 5.0
    --tensor-parallel-size "${TENSOR_PARALLEL}"
    --usp "${ULYSESS_DEGREE}"
    --ring 1
    --vae-patch-parallel-size "${VAE_PATCH_PARALLEL_SIZE}"
    --vae-parallel-mode tile
    --enforce-eager
    --log-stats
)
if [[ "${USE_HSDP}" == "1" ]]; then
    ARGS+=(--use-hsdp)
fi
if [[ "${VAE_USE_TILING}" == "1" ]]; then
    ARGS+=(--vae-use-tiling)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"
# Wan2.2 VAE convs hit MCDNN_STATUS_INVALID_VALUE on MetaX with cudnn on;
# native torch conv fallback works (see cudnn_patch.py).
export VLLM_OMNI_METAX_DISABLE_CUDNN=1

if [[ ! -f "${MODEL}/model_index.json" ]]; then
    echo "error: checkpoint not found at ${MODEL}" >&2
    exit 1
fi

echo "parallel profile: TP=${TENSOR_PARALLEL} USP=${ULYSESS_DEGREE} HSDP=${USE_HSDP} VAE_PP=${VAE_PATCH_PARALLEL_SIZE} VAE_TILING=${VAE_USE_TILING}"
exec vllm serve "${MODEL}" "${ARGS[@]}"
