#!/bin/bash

# Qwen3-ASR e2e server (vllm-omni 0.26 stack; plain vLLM audio support, no --omni).
# Runtime deps (soundfile/librosa/strenum/av) come from the plugin's
# requirements/common.txt — no manual pip steps needed.

vllm serve /mxstorage/pde_ai/models/llm/Qwen/Qwen3-ASR-1.7B/ --served-model-name qwen3-asr-1.7b --disable-log-stats