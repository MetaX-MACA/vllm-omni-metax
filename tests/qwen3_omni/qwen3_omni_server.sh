#!/bin/bash

vllm serve /external/ai/models/llm/Qwen/Qwen3-Omni-30B-A3B-Instruct/ --omni --enforce-eager

#--deploy-config ../../src/vllm_omni_metax/deploy/qwen3_omni_moe.yaml