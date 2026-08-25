#!/bin/bash
# MiniMax H3 Serverless Worker 启动脚本：
# 1) 后台启动 ComfyUI（SageAttention）
# 2) 等待 ComfyUI 就绪
# 3) 运行 RunPod Serverless handler（前台，容器生命周期由它决定）
set -u

COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
LOG=/tmp/comfyui.log

echo "[worker] starting ComfyUI: $COMFY_DIR"
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
