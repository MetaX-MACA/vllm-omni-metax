#!/bin/bash

python openai_speech_client.py --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-Base/ \
    --task-type Base --text "在局子关了三天，出来后非常后悔。" \
    --ref-audio jiayan_zh.wav \
    --ref-text "他当时还跟线下其他的站姐吵架，然后打架进局子了。"

    # --task-type Base --text "在局子关了三天，出来后非常后悔。" \
    # --language Chinese \
    # --ref-audio asr_en.wav \
    # --ref-text "huh.Oh，yeah,yeah.He wasn't even that big when I started listening to him,but and his solo music didn't do overly well,but he did very well when he started writing for other peple"
