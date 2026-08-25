# MiniMax H3 RunPod Serverless Worker
# 基础镜像：本站已实测可跑 MiniMax H3 的 ComfyUI 镜像（CUDA 13.0 / PyTorch 2.10）
FROM runpod/comfyui:cuda13.0

ENV COMFY_DIR=/workspace/runpod-slim/ComfyUI \
    PIP_NO_CACHE_DIR=1

WORKDIR $COMFY_DIR

# 1. 固定 ComfyUI 版本（MiniMax H3 兼容版本）
RUN git fetch --depth=1 origin tag v0.30.2 || git fetch origin tag v0.30.2 \
    && git checkout --detach v0.30.2 \
    && python3 -m pip install -q -r requirements.txt

# 2. 安装 SageAttention 与 Serverless 运行依赖
RUN python3 -m pip install -q -U sageattention requests runpod

# 3. 安装 MiniMax H3 所需的 ComfyUI 自定义节点
RUN mkdir -p "$COMFY_DIR/custom_nodes" && cd "$COMFY_DIR/custom_nodes" \
    && git clone --depth 1 https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git \
    && git clone --depth 1 https://github.com/smthemex/ComfyUI_UniBlockSwap.git \
    && git clone --depth 1 https://github.com/Windecay/ComfyUI-ReservedVRAM.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && for d in */; do if [ -f "$d/requirements.txt" ]; then (cd "$d" && python3 -m pip install -q -r requirements.txt) || echo "WARN requirements failed: $d"; fi; done

# 4. 模型统一从网络卷读取（/runpod-volume/models/...）
COPY extra_model_paths.yaml "$COMFY_DIR/extra_model_paths.yaml"

# 5. 启动脚本与 handler
COPY start.sh /start.sh
RUN chmod +x /start.sh
COPY handler.py /src/handler.py
COPY requirements.txt /src/requirements.txt

WORKDIR /src
CMD ["bash", "/start.sh"]
