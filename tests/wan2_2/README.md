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
2. `patches/wan_sync_patch.py`：Wan2.2 81 帧（21 latent 帧）输出偶发
   灰噪/通道缺失/NaN 的根因是 vllm-omni 阶段流水线在 VAE decode 完成前
   读取解码结果；在 `Wan22Pipeline.forward` 返回前强制 GPU sync 后，
   连续多次 81 帧输出均与参考干净版一致（mean 0.200 / std 0.251）。
3. `patches/cudnn_patch.py`：C600U（sdk3.8.2.8/torch2.8）上 MCDNN conv3d
   对 VAE tiling 的小 tile（32x32）输入计算错误，导致输出边缘色彩模糊/
   色差（tiling vs no-tiling 解码差异 0.63，C500 仅 0.002）。设置
   `VLLM_OMNI_METAX_DISABLE_CUDNN=1` 走 torch 原生卷积后差异降至 0.0023，
   边缘恢复正常。C500 不需要此开关。

> 上述补丁需要随 vllm-omni-metax 重新安装/同步到容器
> （`pip install -e .` 或直接覆盖 site-packages 对应文件）后生效。

## 已知问题（WIP）
- HSDP + USP4 配置下输出存在逐 latent 帧色带/红通道掉色问题（原因未定，
  怀疑序列并行路径）；当前用 TP4 规避。
- 81 帧（21 latent 帧）时，`--vae-patch-parallel-size 4` 的分布式 tile
  解码会稳定出现「重复横条（80px 周期）+ 末尾几帧变绿」：同一份 latent
  用单 rank（`vae-patch-parallel-size 1`）或 plain diffusers 解码均正常，
  问题定位在 vllm-omni 多 rank tile 组装（pack/gather/unpack/merge）。
  已默认改为 `vae-patch-parallel-size 1` 规避；分布式组装待修复。
- 低步数（20 步）+ `flow_shift=5.0`（480p 应为 12.0）时，部分 seed 的
  采样轨迹会坍缩成均匀灰帧（如 seed 65535 + 默认猫拳击 prompt）。
  用 40 步 + `flow_shift=12.0` 后正常；冒烟默认值已改为 seed 12345。
- 画面整体仍偏颗粒感（无平坦区域），与官方 Wan2.2 仓库输出有差距，
  采样器/VAE 方向可继续优化。
