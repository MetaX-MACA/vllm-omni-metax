# MiniMax-H3 测试（vllm-omni-metax）

MiniMax-H3 是 MiniMax 开源的通用全模态生成模型（视频 + 原生立体声音频）。
本目录提供在 MetaX C500 上、基于 `vllm-omni 0.26`（metax 插件）的端到端
冒烟测试：每个用例都要求输出同时包含视频轨和音频轨。

## 覆盖的功能

| 用例 | 分区 | 输入 | 说明 |
| --- | --- | --- | --- |
| T2VA | FL2VA | 文本 | 文生视频 + 音频 |
| FL2VA（首帧） | FL2VA | 文本 + 1 张首帧图 | 首帧图生视频 + 音频 |
| Ref2VA（图 + 音） | Ref2VA | 文本 + 1 张图 + 1 段参考音频 | 参考图像/音色生成视频 + 音频 |
| Ref2VA（视频参考） | Ref2VA | 文本 + 1 段参考视频 | 参考视频延续生成，沿用参考视频原声 |

> 尾帧 / 首尾帧（`frame_indices=[0,-1]`）在当前 metax build 的 HTTP 层不支持：
> pipeline 中 `keyframe_frame_indices` 写死为 `[0]`。官方新版 vllm-omni 才支持，
> 因此本套件不包含该用例。

## 环境与依赖

### 模型权重

两个分区各自独立可服务，目录布局（diffusers 风格）：

```text
<MODEL_ROOT>/
├── FL2VA/    # model_index.json + transformer/tokenizer/processor/text_encoder/video_vae/audio_vae
└── Ref2VA/   # 同上
```

默认 `MODEL_ROOT=/external/ai/share/lli/MiniMax-H3`，可用环境变量覆盖。

### Python 依赖

- `vllm-omni==0.26.0`、`vllm_metax==0.26.0`、`vllm-omni-metax`（本仓库）
- `cache-dit==1.3.0`：vllm-omni 0.26 的多个 diffusion pipeline 会
  `import cache_dit`（MiniMax-H3 在内）。MetaX 运行镜像默认未安装，
  缺了会在启动时报 `No module named 'cache_dit'`。已在
  `requirements/common.txt` 中声明。

### 系统依赖

Ref2VA 视频参考预处理会调用 `ffmpeg` / `ffprobe`：

- `ffprobe`：探测参考视频宽高、帧率、是否有音轨（`_probe_video` / `_has_audio`）
- `ffmpeg`：把参考视频转码为 24fps、目标分辨率、指定帧数（`_transcode_reference_video`）

MetaX 镜像通常自带 `/opt/maca-<ver>/ffmpeg/bin/{ffmpeg,ffprobe}`，但：

1. `ffprobe` 不在 PATH 里；
2. `ffmpeg` 依赖 `libxcb-shm0 / libxcb-shape0 / libxcb-xfixes0`，镜像裁掉了这些库，
   直接运行会报 `error while loading shared libraries: libxcb-shm.so.0`。

修复方式（已整理成 `install_system_deps.sh`）：

```bash
bash tests/minimax_h3/install_system_deps.sh
```

脚本会：apt 安装三个 libxcb 库；若存在 `/opt/maca-*/ffmpeg/bin/ffprobe` 则软链到
`/usr/local/bin/`，否则尝试 `apt-get install -y ffmpeg`。

> 若 `ffmpeg/ffprobe` 缺失，Ref2VA 视频参考请求会在 rank 0 的预处理阶段失败，
> 而 rank 1 卡在 NCCL BROADCAST 上，600 秒后看门狗杀 worker、整个引擎挂掉
> （日志特征：`Watchdog caught collective operation timeout ... OpType=BROADCAST`）。

### 显存要求

- DiT 33B BF16 约 66GB，单张 64GB C500 即使开 CPU offload 也放不下，
  因此两个分区都必须用 **2 张卡**：TP2 + 分布式层卸载（DLO）+ VAE tile。
- 两个分区服务不要同时跑（共享 2 张卡）。

## 快速开始

一键跑全部用例（会依次启动 FL2VA / Ref2VA 服务，全部输出到 `$H3_OUT_DIR`，
默认 `/tmp/h3_tests`）：

```bash
MODEL_ROOT=/external/ai/share/lli/MiniMax-H3 bash tests/minimax_h3/run_all_h3_tests.sh
```

全部通过时输出类似：

```text
PASS t2va                    t2va.mp4                    (vide+soun)
PASS fl2va_first_frame       fl2va_first_frame.mp4       (vide+soun)
PASS ref2va_image_audio      ref2va_image_audio.mp4      (vide+soun)
PASS ref2va_video_ref        ref2va_video_ref.mp4        (vide+soun)
```

## 手动分步方法

### 1. 准备素材

```bash
python3 tests/minimax_h3/make_assets.py /tmp/h3_assets
```

生成 `first_frame.png`、`last_frame.png`（768x768 雪地狐狸场景）和
`ref_audio.wav`（3 秒 32kHz 旋律）。

### 2. 启动 FL2VA 服务（端口 8091）

```bash
H3_PORT=8091 bash tests/minimax_h3/serve_h3_fl2va.sh
```

等待就绪：

```bash
curl -s http://127.0.0.1:8091/v1/models | head -c 200
```

### 3. 跑 T2VA / FL2VA 用例

```bash
OUT=/tmp/h3_tests/t2va.mp4 bash tests/minimax_h3/test_h3_t2va.sh
OUT=/tmp/h3_tests/fl2va_first_frame.mp4 H3_ASSETS_DIR=/tmp/h3_assets \
  bash tests/minimax_h3/test_h3_fl2va.sh
```

### 4. 切换到 Ref2VA 服务（端口 8092）

先停 FL2VA，再启动 Ref2VA：

```bash
kill <FL2VA 服务进程 PID>   # 前台运行按 Ctrl-C 即可
H3_PORT=8092 bash tests/minimax_h3/serve_h3_ref2va.sh
```

### 5. 跑 Ref2VA 用例

```bash
OUT=/tmp/h3_tests/ref2va_image_audio.mp4 H3_ASSETS_DIR=/tmp/h3_assets \
  bash tests/minimax_h3/test_h3_ref2va.sh
```

（该脚本依次执行图 + 音、视频参考两个用例。）

### 6. 校验输出

```bash
python3 tests/minimax_h3/verify_mp4.py /tmp/h3_tests/*.mp4
```

校验逻辑：解析 MP4 box 结构，确认同时存在 `hdlr: vide` 与 `hdlr: soun`，
缺失任一轨时返回非零退出码。

## Web 客户端（页面操作）

在 FL2VA 服务运行的前提下，启动一个 FastAPI 页面客户端，用户直接用浏览器
操作：填写提示词、上传首帧图（可选）、选择分辨率/时长/步数/种子，点“生成”
后页面自动轮询任务状态，完成后直接在线播放并下载 MP4（视频 + 音频）。

```bash
# 方式一：一键脚本（默认后端 127.0.0.1:8091，页面端口 8090）
bash tests/minimax_h3/web_client.sh

# 方式二：直接运行
VLLM_URL=http://127.0.0.1:8091 \
  python3 tests/minimax_h3/web_client.py --host 0.0.0.0 --port 8090
```

浏览器打开（本机为 host 网络，容器内/宿主机均可）：

```text
http://<机器 IP>:8090/
```

页面接口（供脚本/自动化调用）：

| 接口 | 说明 |
| --- | --- |
| `GET /api/health` | 后端 vllm 健康状态与模型 id |
| `POST /api/generate` | 提交任务（multipart：prompt、task=auto/t2va/fl2va、res、duration、steps、flow_shift、seed、image 可选） |
| `GET /api/status/<job_id>` | 查询任务状态（queued/running/done/error） |
| `GET /outputs/<file>` | 下载生成的 MP4 |

当前客户端面向 **FL2VA 分区**（T2VA 文生视频、FL2VA 首帧图生视频）。
若需在页面上支持 Ref2VA（参考图/音频/视频），把 `VLLM_URL` 指向 Ref2VA
服务（`serve_h3_ref2va.sh`，端口 8092），并扩展
`web_client.py` 中的任务类型与 `input_reference`/`input_references`/`audio_reference`
字段（参见上文“API 注意事项”）。

## API 注意事项（当前 build）

- 端点：`POST /v1/videos/sync`（同步，返回 MP4 字节）。
- 表单字段：`prompt / width / height / fps / num_inference_steps /
  flow_shift / seed / extra_params / input_reference / input_references /
  audio_reference`。
- `extra_params` 必须含 `{"task": ..., "duration": ..., "audio_flow_shift": 3.0}`。
- **`input_reference`（单数）**：服务器会解码内容——图片会变成 PIL 帧，视频会变成帧列表。
  用于 FL2VA 首帧、Ref2VA 图片参考。
- **`input_references`（复数）**：服务器会把上传文件**落盘成临时文件路径**再传给
  pipeline。H3 的视频参考预处理只接受文件路径，所以 **Ref2VA 视频参考必须用
  `input_references`**；用单数 `input_reference` 传视频会报
  `MiniMax H3 multi-video Ref2VA currently requires file paths`。
- `audio_reference` 接受 `{"audio_url": "<http(s) url 或 data:audio/wav;base64,...>"}`。
  base64 较大时不要把 data URL 直接拼进命令行（会 `Argument list too long`），
  先写入 JSON 文件再用 `-F 'audio_reference=<file.json'`（本套件已这么做）。

## 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 启动报 `No module named 'cache_dit'` | 未装 `cache-dit==1.3.0`；`pip install cache-dit==1.3.0`（已加入 requirements/common.txt） |
| 视频参考请求卡 10 分钟后引擎挂掉（NCCL BROADCAST 超时） | `ffmpeg/ffprobe` 缺失；执行 `install_system_deps.sh` 后重启服务 |
| `ffmpeg: error while loading shared libraries: libxcb-shm.so.0` | 缺 libxcb；`apt-get install -y libxcb-shm0 libxcb-shape0 libxcb-xfixes0` |
| `curl: Argument list too long` | audio data URL 太大；改用 `-F 'audio_reference=<file'` |
| 图 + 音用例报 `ref2va requires multi_modal_data.image or video` | 检查是否传了 `input_reference`（单数）且任务为 `ref2va` |
