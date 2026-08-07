#!/bin/bash

# CustomVoice model: use task_type=CustomVoice with a built-in speaker
# (aiden/vivian/serena/ryan/uncle_fu/ono_anna/sohee/eric/dylan).
# For voice cloning (task_type=Base) serve Qwen3-TTS-12Hz-1.7B-Base instead.
vllm-omni serve /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/ \
    --omni \
    --trust-remote-code \
    -tp 1 \
    --max-model-len 8192 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --port 8500
