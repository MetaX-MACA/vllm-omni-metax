#!/bin/bash

vllm-omni serve /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-Base/ \
    --omni \
    --trust-remote-code
