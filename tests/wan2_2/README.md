# Wan2.2-T2V 测试（vllm-omni-metax）

在 MetaX C500 上、基于 `vllm-omni 0.26`（metax 插件）跑
`Wan-AI/Wan2.2-T2V-A14B-Diffusers` 文生视频的端到端冒烟测试。

## 模型与硬件

- 模型：`/mxstorage/pde_ai/models/llm/Wan-AI/Wan2.2-T2V-A14B-Diffusers`
  （Diffusers 格式，transformer + transformer_2 双段 MoE，约 66 GB BF16）
- 硬件：4 x MetaX C500（64 GB/卡）
- 容器：`vllm_omni_lli_original`（vllm 0.26.0 + vllm-omni-metax）

## 启动服务

```bash
# 默认 TP4（推荐，已验证可跑通）
bash tests/wan2_2/run_wan22_t2v.sh

# 备选：HSDP + USP4（能启动，但当前 MetaX 栈输出有色带/闪烁问题，勿用于正式结果）
USE_HSDP=1 ULYSESS_DEGREE=4 TENSOR_PARALLEL=1 bash tests/wan2_2/run_wan22_t2v.sh
```

服务监听 `0.0.0.0:8091`（host 网络）。模型加载约 5 分钟，
日志可用 `--log-stats` 观察；健康检查 `curl http://127.0.0.1:8091/health`。

## 发请求

```bash
# 冒烟（480x832 / 49 帧 / 20 步）
bash tests/wan2_2/test_wan22_t2v.sh

# 全分辨率效果（720x1280 / 81 帧 / 40 步 / 480p 对应 flow_shift=12）
OUT=/tmp/wan22_hd.mp4 HEIGHT=720 WIDTH=1280 NUM_FRAMES=81 \
NUM_INFERENCE_STEPS=40 FLOW_SHIFT=12.0 FPS=24 \
bash tests/wan2_2/test_wan22_t2v.sh
```

端点：`POST /v1/videos/sync`（同步返回 mp4）；`POST /v1/videos`（异步任务）。
参数：`prompt / width / height / num_frames / fps / num_inference_steps /
guidance_scale / guidance_scale_2 / boundary_ratio / flow_shift / seed /
negative_prompt / extra_params`。

## 本目录涉及的环境修复（已合入 vllm-omni-metax）

1. `patches/rope_patch.py`：修复 rotary shim 的 NameError /
   UnboundLocalError（原代码调用未定义的 `ext_apply_rotary_emb`，且 `ext_fn`
   是局部变量第二次调用即失效；改为模块级缓存 `_EXT_ROTARY_FN`）。
2. `patches/cudnn_patch.py`（新增，`plugin.py` 注册）：Wan2.2 VAE 卷积在
   MetaX 上报 `MCDNN_STATUS_INVALID_VALUE`，通过
   `VLLM_OMNI_METAX_DISABLE_CUDNN=1` 关闭 cudnn、走 torch 原生卷积解决。

> 上述补丁需要随 vllm-omni-metax 重新安装/同步到容器
> （`pip install -e .` 或直接覆盖 site-packages 对应文件）后生效。

## 已知问题（WIP）

- HSDP + USP4 配置下输出存在逐 latent 帧色带/红通道掉色问题（原因未定，
  怀疑序列并行路径）；当前用 TP4 规避。
- TP4 输出颜色均衡、无拼接缝，但整体仍偏“低对比度 + 颗粒感”，
  与官方 Wan2.2 仓库输出（明暗结构明显）有差距，仍在排查（采样/精度/CFG 方向）。
