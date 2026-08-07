#!/bin/bash

python openai_speech_client.py --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-Base/ \
    --task-type Base --text "hello,this is a test voice" \
    --ref-audio asr_en.wav \
    --ref-text "huh.Oh，yeah,yeah.He wasn't even that big when I started listening to him,but and his solo music didn't do overly well,but he did very well when he started writing for other peple"