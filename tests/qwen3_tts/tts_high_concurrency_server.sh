#!/bin/bash

# NEW 0.26-era verification: Qwen3-TTS high-concurrency profile using the
# builtin vllm_omni/deploy/qwen3_tts_high_concurrency.yaml (2-GPU, stage 0
# on GPU 0 with S0=64, stage 1 on GPU 1 with S1=10; codec_streaming,
# code_predictor_prefix_graphs, ref_code_context_frames, chunk ramp).
# The bare name resolves against the installed vllm_omni/deploy/ dir.
#
# Drive with 64 concurrent clients (see tts_client.sh / openai_speech_client.py)
# and check RTF/TTFA plus chunk-streaming behaviour.

vllm serve /external/ai/models/llm/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base/ \
    --omni \
    --trust-remote-code \
    --enforce-eager \
    --deploy-config qwen3_tts_high_concurrency.yaml
