import base64
import json
import urllib.request

#huh.Oh，yeah,yeah.He wasn't even that big when I started listening to him,but and his solo music didn't do overly well,but he did very well when he started writing for other peple
with open("asr_en.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                }
            ]
        }
    ]
}

req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req) as response:
    print(response.read().decode("utf-8"))