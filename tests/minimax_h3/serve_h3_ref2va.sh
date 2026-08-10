#!/usr/bin/env bash
# MiniMax-H3 Ref2VA server (vllm-omni 0.26 on MetaX C500).
#
# Same validated 2-GPU profile as serve_h3_fl2va.sh.  Run one partition at a
# time: FL2VA and Ref2VA each need the same 2 GPUs.
#
# Env overrides:
#   MODEL_ROOT   model root containing FL2VA/ and Ref2VA/  (default below)
#   H3_PORT      server port                              (default 8092)
#   H3_DEVICES   CUDA_VISIBLE_DEVICES                     (default 0,1)
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/external/ai/share/lli/MiniMax-H3}"
export PORT="${H3_PORT:-8092}"
export CUDA_VISIBLE_DEVICES="${H3_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"

MODEL="${MODEL_ROOT}/Ref2VA"
if [[ ! -f "${MODEL}/model_index.json" ]]; then
    echo "error: Ref2VA checkpoint not found at ${MODEL}" >&2
    exit 1
fi

exec vllm serve "${MODEL}" \
    --omni \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --num-gpus 2 \
    --tensor-parallel-size 2 \
    --usp 1 \
    --ring 1 \
    --text-encoder-tp-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --enforce-eager
