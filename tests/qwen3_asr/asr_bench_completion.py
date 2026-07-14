#!/usr/bin/env python3
"""
离线 ASR benchmark：用 /v1/chat/completions + base64 audio 格式压测 Qwen3-ASR。
不依赖 vllm bench，直接从缓存 arrow 文件读数据。

用法:
    python bench_asr.py --num-prompts 100 --concurrency 8
    python bench_asr.py --num-prompts 500 --concurrency 32 --duration-ms 60000
"""
import os, sys, json, time, io, base64, asyncio, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow as pa
import soundfile as sf
import aiohttp

# ── 配置 ──────────────────────────────────────────────────
SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://127.0.0.1:8000")
CACHE_ROOT = Path(os.path.expanduser("~/.cache/huggingface/datasets"))
DATASET_ID = "openslr/librispeech_asr"
DATASET_CACHE_NAME = "openslr___librispeech_asr"

# 缓存 hash 目录（与 build_cache_local.py 一致）
CACHE_HASH = os.environ.get("CACHE_HASH", "191e0692c57540f2e6d90b59ddb02a3f")
CACHE_DIR = CACHE_ROOT / DATASET_CACHE_NAME / "clean" / "0.0.0" / CACHE_HASH


def load_audio_samples(split: str = "test", max_samples: int = 1000):
    """从缓存 arrow 文件加载音频样本。"""
    arrow_file = CACHE_DIR / f"librispeech_asr-{split}.arrow"
    if not arrow_file.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_file}\n"
                                f"Run build_cache_local.py --build --splits {split} first.")

    with pa.ipc.open_stream(str(arrow_file)) as reader:
        tbl = reader.read_all()

    samples = []
    audio_col = tbl.column("audio")
    text_col = tbl.column("text")
    n = min(len(tbl), max_samples)

    for i in range(n):
        audio = audio_col[i].as_py()
        samples.append({
            "array": np.asarray(audio["array"], dtype=np.float32)
                     if not isinstance(audio["array"], np.ndarray) else audio["array"],
            "sr": audio["sampling_rate"],
            "text": text_col[i].as_py(),
        })

    return samples


def make_payload(audio_array: np.ndarray, sr: int, model: str = "qwen3-asr-1.7b") -> dict:
    """构造 /v1/chat/completions 请求 payload（base64 WAV 格式）。"""
    buf = io.BytesIO()
    sf.write(buf, audio_array, sr, format="WAV")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {
                    "data": audio_b64,
                    "format": "wav",
                }},
                {"type": "text", "text": "Transcribe the audio."},
            ],
        }],
        "max_completion_tokens": 256,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def send_one(session: aiohttp.ClientSession, payload: dict,
                   sample_idx: int, stats: list):
    """发送单个流式请求，记录 TTFT / TPOT / latency。"""
    url = f"{SERVER_URL}/v1/chat/completions"
    start = time.perf_counter()
    ttft = 0.0
    last_ts = start
    output_tokens = 0
    inter_token_lats = []

    try:
        async with session.post(url, json=payload,
                                headers={"Content-Type": "application/json"}) as resp:
            if resp.status != 200:
                body = await resp.text()
                stats.append({"error": f"HTTP {resp.status}: {body[:200]}", "idx": sample_idx})
                return

            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    now = time.perf_counter()
                    if ttft == 0.0:
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                            ttft = now - start
                            last_ts = now
                    else:
                        inter_token_lats.append(now - last_ts)
                        last_ts = now

                    if chunk.get("usage"):
                        output_tokens = chunk["usage"].get("completion_tokens", 0)

        elapsed = time.perf_counter() - start
        stats.append({
            "idx": sample_idx,
            "ttft": ttft * 1000,
            "tpot": (sum(inter_token_lats) / len(inter_token_lats) * 1000) if inter_token_lats else 0,
            "latency": elapsed * 1000,
            "output_tokens": output_tokens,
            "itl": [x * 1000 for x in inter_token_lats],
        })

    except Exception as e:
        stats.append({"error": str(e)[:200], "idx": sample_idx})


async def run_bench(samples: list, concurrency: int, num_prompts: int,
                    duration_ms: int = 0):
    """并发压测主循环。"""
    stats: list = []
    sem = asyncio.Semaphore(concurrency)

    async def worker(payload, idx):
        async with sem:
            await send_one(session, payload, idx, stats)

    connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        start_time = time.perf_counter()
        deadline = start_time + duration_ms / 1000 if duration_ms else float("inf")

        for i in range(num_prompts):
            sample = samples[i % len(samples)]
            payload = make_payload(sample["array"], sample["sr"])
            tasks.append(asyncio.create_task(worker(payload, i)))

            if duration_ms and time.perf_counter() > deadline:
                break

        print(f"Running {len(tasks)} requests with concurrency={concurrency}...")
        await asyncio.gather(*tasks)

    return stats


def report(stats: list, total_dur_s: float):
    """打印 benchmark 报告。"""
    ok = [s for s in stats if "error" not in s]
    err = [s for s in stats if "error" in s]
    print(f"\n{'='*60}")
    print(f"Requests: {len(stats)} total, {len(ok)} OK, {len(err)} failed")
    print(f"Duration: {total_dur_s:.1f}s")
    print(f"Throughput: {len(ok)/total_dur_s:.2f} req/s")

    if not ok:
        if err:
            print(f"Errors: {err[:3]}")
        return

    ttfts = [s["ttft"] for s in ok if s["ttft"] > 0]
    tpots = [s["tpot"] for s in ok if s["tpot"] > 0]
    latencies = [s["latency"] for s in ok]
    all_itl = [x for s in ok for x in s.get("itl", [])]

    def pct(data, p):
        return np.percentile(data, p) if data else 0

    print(f"\n{'Metric':<15} {'Avg':>8} {'Min':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, data in [("TTFT (ms)", ttfts), ("TPOT (ms)", tpots),
                        ("Latency (ms)", latencies), ("ITL (ms)", all_itl)]:
        if data:
            print(f"{name:<15} {np.mean(data):8.1f} {np.min(data):8.1f} "
                  f"{pct(data,50):8.1f} {pct(data,95):8.1f} "
                  f"{pct(data,99):8.1f} {np.max(data):8.1f}")
    print(f"\nOutput tokens avg: {np.mean([s['output_tokens'] for s in ok]):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Benchmark via /v1/chat/completions")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--duration-ms", type=int, default=0,
                        help="Max duration in ms (0 = unlimited)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--server", type=str, default=SERVER_URL,
                        help="vLLM server URL")
    args = parser.parse_args()

    SERVER_URL = args.server.rstrip("/")

    print(f"Server: {SERVER_URL}")
    print(f"Loading audio samples from {CACHE_DIR}...")
    samples = load_audio_samples(split=args.split, max_samples=args.num_prompts)
    print(f"Loaded {len(samples)} samples "
          f"(avg {np.mean([len(s['array'])/s['sr'] for s in samples]):.1f}s each)")

    t0 = time.perf_counter()
    stats = asyncio.run(run_bench(samples, args.concurrency, args.num_prompts,
                                  args.duration_ms))
    total_dur = time.perf_counter() - t0

    report(stats, total_dur)
