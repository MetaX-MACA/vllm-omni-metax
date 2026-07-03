#!/bin/bash

python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image --model "/external/ai/models/llm/Qwen/Qwen3-Omni-30B-A3B-Instruct/" --image-path "dog.jpg" \
    --prompt "描述一下这张图片"
    # --stream

# 安装 sox
# sudo apt install sox libsox-fmt-all

# 合并（自动按字母/数字排序）
# sox audio_chatcmpl-8f97742a54a59029_*.wav output.wav