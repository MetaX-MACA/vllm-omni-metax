#!/bin/bash

# NEW 0.26-era verification: streaming TTS word timestamps via the forced
# aligner. --forced-aligner overrides the model field of the aligner YAML;
# --forced-aligner-config points at the builtin reference YAML
# (vllm_omni/deploy/qwen3_tts_forced_aligner.yaml).
#
# Assert: /v1/audio/speech responses carry word timestamps when
# word_timestamps=true is requested.

vllm serve /external/ai/models/llm/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base/ \
    --omni \
    --trust-remote-code \
    --enforce-eager \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B \
    --forced-aligner-config "$(python -c 'import os, vllm_omni; print(os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_tts_forced_aligner.yaml"))')"
