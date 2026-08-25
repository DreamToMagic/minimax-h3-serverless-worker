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

# 如果 ComfyUI 已在运行则复用（镜像可能自带启动逻辑）
if ! curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  nohup python3 main.py --listen 127.0.0.1 --port 8188 --enable-cors-header --use-sage-attention > "$LOG" 2>&1 &
fi

READY=0
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo "[worker] ComfyUI ready after ${i} checks"
    READY=1
    break
  fi
  if [ $((i % 15)) -eq 0 ]; then
    echo "[worker] ComfyUI not ready yet (${i}), last log:"; tail -5 "$LOG" 2>/dev/null || true
  fi
  sleep 2
done

if [ "$READY" != "1" ]; then
  echo "[worker] FATAL: ComfyUI failed to start"
  tail -40 "$LOG" 2>/dev/null || true
  exit 1
fi

echo "[worker] attention log:"
grep -i -E 'sage|attention' "$LOG" | head -3 || true

echo "[worker] starting handler"
cd /src || exit 1
exec python3 -u handler.py
