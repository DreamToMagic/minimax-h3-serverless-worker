#!/bin/bash
# MiniMax H3 Serverless Worker 启动脚本：
# 1) 后台启动 ComfyUI（SageAttention）
# 2) 等待 ComfyUI 就绪
# 3) 运行 RunPod Serverless handler（前台，容器生命周期由它决定）
set -u

COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
LOG=/tmp/comfyui.log

# 基础镜像在不同运行环境里 ComfyUI 位置可能不同：先探测 main.py 的真实位置
if [ ! -f "$COMFY_DIR/main.py" ]; then
  echo "[worker] $COMFY_DIR/main.py not found, searching..."
  FOUND=$(find /workspace /opt /app /root /src -maxdepth 7 -type f -name main.py 2>/dev/null | grep -i comfy | head -1)
  if [ -n "$FOUND" ]; then
    COMFY_DIR=$(dirname "$FOUND")
    echo "[worker] found ComfyUI at $COMFY_DIR"
  fi
fi

# 兜底：镜像里确实没有 ComfyUI 源码时，直接拉 v0.30.2
if [ ! -f "$COMFY_DIR/main.py" ]; then
  echo "[worker] cloning ComfyUI v0.30.2 as fallback"
  git clone --depth 1 --branch v0.30.2 https://github.com/comfyanonymous/ComfyUI.git /src/ComfyUI || exit 1
  COMFY_DIR=/src/ComfyUI
  python3 -m pip install -q -r "$COMFY_DIR/requirements.txt" || echo "WARN: requirements failed"
  # 把已构建进镜像的自定义节点搬过来
  if [ -d /workspace/runpod-slim/ComfyUI/custom_nodes ]; then
    cp -a /workspace/runpod-slim/ComfyUI/custom_nodes/. "$COMFY_DIR/custom_nodes/" 2>/dev/null || true
  fi
  cat > "$COMFY_DIR/extra_model_paths.yaml" <<'YAML'
minimax_h3_volume:
  base_path: /runpod-volume/models
  checkpoints: diffusion_models
  diffusion_models: diffusion_models
  unet: diffusion_models
  clip: text_encoders
  text_encoders: text_encoders
  vae: vae
  loras: loras
YAML
fi

echo "[worker] ComfyUI dir: $COMFY_DIR"
ls -la "$COMFY_DIR" | head -30 || true

cd "$COMFY_DIR" || exit 1

# 清掉基础镜像可能已在运行的 ComfyUI，避免端口占用/进程混乱
pkill -f 'python3 main.py' 2>/dev/null || true
pkill -f 'python main.py' 2>/dev/null || true
sleep 2

# 后台启动 ComfyUI；日志同时写入 /tmp/comfyui.log 和 stdout。
# 注意：这里绝不能等待 ComfyUI 就绪——RunPod 要求 handler 在约 1 分钟内
# 监听 8000 端口，等待会触发平台把 worker 判死。ComfyUI 的就绪检查
# 推迟到 handler 收到第一个任务时进行。
( python3 main.py --listen 127.0.0.1 --port 8188 --enable-cors-header --use-sage-attention 2>&1 | tee -a "$LOG" ) &

echo "[worker] starting handler"
cd /src || exit 1
exec python3 -u handler.py
