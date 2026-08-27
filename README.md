# MiniMax H3 RunPod Serverless Worker

多用户平台的按需 GPU 生成 Worker。容器内运行 ComfyUI v0.30.2 + MiniMax H3
自定义节点，接收生成参数，生成完成后把 mp4 回传到平台 webhook。

## 架构

```
平台 /api/user/generate
   → RunPod Serverless Endpoint /run
   → 本容器 handler.py（ComfyUI 生成 + 上传回传）
   → 平台 webhook 落盘到用户 outputs
```

## 模型/加速说明

- 默认使用 `pruned INT8`。
- **默认不加载 Turbo LoRA**；只有任务显式传入 `turbo_lora: true` 时才启用，避免在 8 步以上产生过度平滑/果冻感。

## 文件

- `Dockerfile`：基于 `runpod/comfyui:cuda13.0`（本站已实测），固定
  ComfyUI v0.30.2，安装 SageAttention + T8 等自定义节点
- `start.sh`：启动 ComfyUI（`--use-sage-attention`）并等待就绪后运行 handler
- `handler.py`：RunPod Serverless handler；启动自检（CUDA / 模型文件 /
  ComfyUI health），构造 T2V graph，轮询生成结果，上传平台 webhook
- `extra_model_paths.yaml`：模型统一从网络卷 `/runpod-volume/models/` 读取

## 镜像

GitHub Actions 自动构建并推送到：

```text
ghcr.io/dreamtomagic/minimax-h3-serverless-worker:latest
```

## 网络卷目录要求

```text
/runpod-volume/models/
├── diffusion_models/
│   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors   # 默认必需
│   └── minimax_h3_fl2va_int8_convrot.safetensors          # 可选（完整版，已从卷删除）
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
├── vae/
│   ├── minimax_h3_video_vae_fp16.safetensors
│   └── minimax_h3_audio_vae_fp32.safetensors
└── loras/
    └── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors  # 仅 turbo_lora=true 时必需
```

## 环境变量（Serverless 模板配置）

| 变量 | 说明 |
|---|---|
| `WEBHOOK_URL` | `https://dreamtomagic.top/api/user/generate-webhook` |
| `WEBHOOK_SECRET` | 与平台 `/root/.minimax_webhook_secret` 一致 |
| `COMFY_URL` | 默认 `http://127.0.0.1:8188` |
| `GENERATION_TIMEOUT` | 默认 2400 秒 |
