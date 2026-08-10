#!/usr/bin/env python3
"""MiniMax-H3 web client for the local vllm-omni server (FL2VA partition).

The page lets a user generate video+audio from text (T2VA) or from a first
frame image (FL2VA).  The client proxies the request to the vllm-omni
``POST /v1/videos/sync`` endpoint in a background job so the browser does not
hold a multi-minute HTTP request; the page polls ``/api/status/<job_id>``.

Run:
    VLLM_URL=http://127.0.0.1:8091 python3 web_client.py --port 8090

Then open http://<host>:8090/ in a browser.
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8091")
OUTPUT_DIR = Path(os.environ.get("H3_WEB_OUTPUT_DIR", "/tmp/h3_web_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MiniMax-H3 Web Client")

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniMax-H3 生成客户端</title>
<style>
  :root { --bg:#0f1420; --card:#171e2e; --line:#2a3448; --fg:#e8ecf4;
          --dim:#8fa0b8; --acc:#4f8cff; --ok:#37d67a; --err:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.6 "PingFang SC","Microsoft YaHei",system-ui,sans-serif; }
  .wrap { max-width:920px; margin:0 auto; padding:28px 18px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--dim); margin-bottom:20px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:14px; padding:20px; margin-bottom:18px; }
  label { display:block; color:var(--dim); font-size:13px; margin:0 0 6px; }
  textarea, select, input[type=number] {
      width:100%; background:#0e1420; color:var(--fg);
      border:1px solid var(--line); border-radius:8px;
      padding:10px 12px; font:inherit; outline:none; }
  textarea:focus, select:focus, input:focus { border-color:var(--acc); }
  textarea { min-height:90px; resize:vertical; }
  .row { display:flex; gap:14px; flex-wrap:wrap; }
  .row > div { flex:1; min-width:150px; }
  .btn { background:var(--acc); color:#fff; border:0; border-radius:10px;
         padding:12px 22px; font-size:16px; cursor:pointer; width:100%; }
  .btn:disabled { opacity:.45; cursor:not-allowed; }
  .badge { display:inline-block; padding:2px 10px; border-radius:99px;
           font-size:12px; margin-left:8px; }
  .badge.ok { background:rgba(55,214,122,.15); color:var(--ok); }
  .badge.err { background:rgba(255,107,107,.15); color:var(--err); }
  .badge.warn { background:rgba(255,190,80,.15); color:#ffbe50; }
  #drop { border:1.5px dashed var(--line); border-radius:10px; padding:16px;
          text-align:center; color:var(--dim); cursor:pointer; }
  #drop.hot { border-color:var(--acc); background:rgba(79,140,255,.08); }
  #prev { max-width:180px; max-height:120px; border-radius:8px;
          margin-top:10px; display:none; }
  video { width:100%; border-radius:10px; background:#000; }
  #status { display:none; margin-top:12px; }
  #status .msg { color:var(--dim); }
  #err { display:none; color:var(--err); margin-top:12px;
         white-space:pre-wrap; }
  a.dl { display:inline-block; margin-top:10px; color:var(--acc); }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:640px){ .grid2{ grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>MiniMax-H3 生成客户端 <span id="be" class="badge warn">连接中…</span></h1>
  <div class="sub">基于 vllm-omni（FL2VA 分区）· 输出视频 + 原生立体声音频 ·
     端口 8091 后端</div>

  <div class="card">
    <label>任务类型</label>
    <select id="task">
      <option value="auto">自动判断（上传图片=FL2VA 首帧，否则=T2VA 文生视频）</option>
      <option value="t2va">T2VA 文生视频（无需图片）</option>
      <option value="fl2va">FL2VA 首帧图生视频（需上传 1 张图）</option>
    </select>
  </div>

  <div class="card">
    <label>提示词（描述画面与声音）</label>
    <textarea id="prompt"
      placeholder="例如：一只小狐狸在雪后的蓝紫色森林里漫步，雪地传来脚步声，远处有鸟鸣。"></textarea>

    <div class="grid2" style="margin-top:14px;">
      <div>
        <label>首帧图片（可选）</label>
        <div id="drop">点击或拖拽上传图片</div>
        <input type="file" id="file" accept="image/*" style="display:none">
        <img id="prev" alt="preview">
      </div>
      <div>
        <label>输出分辨率</label>
        <select id="res">
          <option value="1024x576">1024x576 (16:9)</option>
          <option value="768x768">768x768 (1:1)</option>
          <option value="1280x720">1280x720 (16:9)</option>
          <option value="auto">自动（跟随图片，短边 768）</option>
        </select>
        <label style="margin-top:12px;">时长（秒，4–15）</label>
        <select id="duration">
          <option>4</option><option selected>5</option><option>8</option>
          <option>10</option><option>15</option>
        </select>
      </div>
    </div>

    <div class="row" style="margin-top:14px;">
      <div><label>去噪步数（1–50）</label><input type="number" id="steps" value="20" min="1" max="50"></div>
      <div><label>随机种子（留空=随机）</label><input type="number" id="seed" placeholder="随机"></div>
      <div><label>flow_shift</label><input type="number" id="flow_shift" value="12" step="0.5"></div>
    </div>

    <div style="margin-top:16px;">
      <button class="btn" id="go">生成视频 + 音频</button>
    </div>

    <div id="status">
      <div class="msg" id="msg"></div>
    </div>
    <div id="err"></div>
  </div>

  <div class="card" id="resultCard" style="display:none;">
    <label>生成结果</label>
    <video id="player" controls playsinline></video>
    <a class="dl" id="dl" download>下载 MP4</a>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let imgFile = null, pollTimer = null;

async function health() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    if (d.vllm_ok) {
      $('be').textContent = '后端就绪 · ' + (d.model || '');
      $('be').className = 'badge ok';
    } else {
      $('be').textContent = '后端未就绪';
      $('be').className = 'badge err';
    }
  } catch (e) {
    $('be').textContent = '连接失败';
    $('be').className = 'badge err';
  }
}

$('drop').addEventListener('click', () => $('file').click());
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').classList.add('hot'); });
$('drop').addEventListener('dragleave', () => $('drop').classList.remove('hot'));
$('drop').addEventListener('drop', e => {
  e.preventDefault(); $('drop').classList.remove('hot');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
$('file').addEventListener('change', e => { if (e.target.files.length) setFile(e.target.files[0]); });

function setFile(f) {
  imgFile = f;
  $('prev').src = URL.createObjectURL(f);
  $('prev').style.display = 'block';
  $('drop').textContent = f.name;
}

$('go').addEventListener('click', async () => {
  const prompt = $('prompt').value.trim();
  if (!prompt) { $('err').textContent = '请先填写提示词'; $('err').style.display = 'block'; return; }
  $('err').style.display = 'none';
  $('resultCard').style.display = 'none';
  $('go').disabled = true;
  $('status').style.display = 'block';
  $('msg').textContent = '已提交，正在排队…';

  const fd = new FormData();
  fd.append('prompt', prompt);
  fd.append('task', $('task').value);
  fd.append('res', $('res').value);
  fd.append('duration', $('duration').value);
  fd.append('steps', $('steps').value);
  fd.append('flow_shift', $('flow_shift').value);
  if ($('seed').value) fd.append('seed', $('seed').value);
  if (imgFile) fd.append('image', imgFile);

  try {
    const r = await fetch('/api/generate', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || d.error || '提交失败');
    const jobId = d.job_id;
    const t0 = Date.now();
    $('msg').textContent = '生成中…（20 步约 3–6 分钟，请耐心等待）';
    pollTimer = setInterval(async () => {
      try {
        const s = await (await fetch('/api/status/' + jobId)).json();
        if (s.status === 'running') {
          const sec = Math.round((Date.now() - t0) / 1000);
          $('msg').textContent = '生成中… 已用时 ' + sec + ' 秒';
        } else if (s.status === 'done') {
          clearInterval(pollTimer);
          $('msg').textContent = '完成，用时 ' + Math.round(s.elapsed) + ' 秒';
          $('player').src = s.output_url;
          $('dl').href = s.output_url;
          $('resultCard').style.display = 'block';
          $('go').disabled = false;
        } else {
          clearInterval(pollTimer);
          $('msg').textContent = '失败';
          $('err').textContent = s.error || '生成失败';
          $('err').style.display = 'block';
          $('go').disabled = false;
        }
      } catch (e) { /* transient */ }
    }, 3000);
  } catch (e) {
    $('err').textContent = String(e.message || e);
    $('err').style.display = 'block';
    $('go').disabled = false;
    $('status').style.display = 'none';
  }
});

health();
setInterval(health, 30000);
</script>
</body>
</html>
"""


def _vllm_extra_params(task: str, duration: float) -> str:
    return '{"task":"%s","duration":%s,"audio_flow_shift":3.0}' % (task, duration)


def _run_job(job_id: str, payload: dict) -> None:
    started = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "running"
    try:
        data = {
            "prompt": payload["prompt"],
            "fps": "24",
            "num_inference_steps": str(payload["steps"]),
            "flow_shift": str(payload["flow_shift"]),
            "seed": str(payload["seed"]),
            "extra_params": _vllm_extra_params(payload["task"], payload["duration"]),
        }
        if payload.get("width"):
            data["width"] = str(payload["width"])
            data["height"] = str(payload["height"])
        files = None
        if payload.get("image_bytes"):
            files = {"input_reference": ("first_frame.png", payload["image_bytes"], "image/png")}
        with httpx.Client(timeout=7200) as client:
            resp = client.post(f"{VLLM_URL}/v1/videos/sync", data=data, files=files)
        if resp.status_code != 200:
            raise RuntimeError(f"vllm returned HTTP {resp.status_code}: {resp.text[:500]}")
        out_name = f"h3_{payload['task']}_{job_id[:8]}.mp4"
        out_path = OUTPUT_DIR / out_name
        out_path.write_bytes(resp.content)
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status="done",
                elapsed=round(time.time() - started, 1),
                size=len(resp.content),
                output_url=f"/outputs/{out_name}",
                output_path=str(out_path),
            )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the page
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error", error=str(exc), elapsed=round(time.time() - started, 1))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/health")
def health() -> dict:
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{VLLM_URL}/v1/models")
            model = r.json()["data"][0]["id"] if r.status_code == 200 and r.json().get("data") else None
        return {"vllm_ok": r.status_code == 200, "model": model, "vllm_url": VLLM_URL}
    except Exception as exc:  # noqa: BLE001
        return {"vllm_ok": False, "error": str(exc), "vllm_url": VLLM_URL}


@app.post("/api/generate")
async def generate(
    prompt: str = Form(...),
    task: str = Form("auto"),
    res: str = Form("1024x576"),
    duration: float = Form(5.0),
    steps: int = Form(20),
    flow_shift: float = Form(12.0),
    seed: int | None = Form(None),
    image: UploadFile | None = File(None),
) -> JSONResponse:
    if task == "auto":
        task = "fl2va" if image is not None else "t2va"
    if task not in ("t2va", "fl2va"):
        return JSONResponse({"detail": f"当前 FL2VA 分区仅支持 t2va / fl2va，收到 {task}"}, status_code=400)
    duration = min(15.0, max(4.0, float(duration)))
    steps = min(50, max(1, int(steps)))
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    width = height = None
    if res and res != "auto":
        try:
            width, height = (int(v) for v in res.lower().split("x"))
        except ValueError:
            width = height = None

    image_bytes = await image.read() if image is not None else None
    job_id = uuid.uuid4().hex
    payload = {
        "prompt": prompt,
        "task": task,
        "duration": duration,
        "steps": steps,
        "flow_shift": flow_shift,
        "seed": seed,
        "width": width,
        "height": height,
        "image_bytes": image_bytes,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "queued", "task": task, "created": time.time()}
    threading.Thread(target=_run_job, args=(job_id, payload), daemon=True).start()
    return {"job_id": job_id, "task": task}


@app.get("/api/status/{job_id}")
def status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return {"status": "error", "error": "job not found"}
    return job


@app.get("/outputs/{name}")
def output(name: str) -> FileResponse:
    path = OUTPUT_DIR / name
    if not path.is_file():
        return FileResponse(status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    print(f"MiniMax-H3 web client: http://{args.host}:{args.port}  (vllm: {VLLM_URL})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
