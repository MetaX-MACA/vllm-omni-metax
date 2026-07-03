#!/bin/bash

vllm-omni serve /external/ai/models/llm/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base/ \
    --omni \
    --trust-remote-code \
    --enforce-eager \
