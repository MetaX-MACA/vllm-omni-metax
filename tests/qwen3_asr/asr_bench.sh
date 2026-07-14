#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

vllm bench serve \
    --backend openai-audio \
    --model /external/ai/models/llm/Qwen/Qwen3-ASR-1.7B/ \
    --endpoint /v1/audio/transcriptions \
    --dataset-name hf \
    --dataset-path openslr/librispeech_asr \
    --hf-subset clean \
    --hf-split test \
    --num-prompts 10
    # --hf-split validation
