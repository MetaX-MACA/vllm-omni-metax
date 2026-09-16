#!/bin/bash

# Qwen3-TTS high-concurrency server profile tuned for MetaX (MACA).
# Same launch shape as tts_server.sh, but with the MetaX-adapted
# qwen3_tts_high_concurrency_metax.yaml (stage1 max_num_seqs=12,
# decode_batch_max_size=12; inner Code2Wav CUDA graph is a no-op on MetaX).

vllm serve /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-Base/ \
    --omni \
    --trust-remote-code \
    --deploy-config src/vllm_omni_metax/deploy/qwen3_tts_high_concurrency_metax.yaml
