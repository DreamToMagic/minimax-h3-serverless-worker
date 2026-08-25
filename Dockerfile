# MiniMax H3 RunPod Serverless Worker
# 基础镜像：本站已实测可跑 MiniMax H3 的 ComfyUI 镜像（CUDA 13.0 / PyTorch 2.10）
FROM runpod/comfyui:cuda13.0

# 基础镜像里 ComfyUI 的真实位置是 /opt/comfyui-baked
# （/workspace/runpod-slim/ComfyUI 只是 Pod 模式用的空挂载点）
ENV COMFY_DIR=/opt/comfyui-baked \
    PIP_NO_CACHE_DIR=1

WORKDIR $COMFY_DIR

# 1. 按镜像自带 requirements.txt 升级依赖（MiniMax H3 需要的
#    comfy-kitchen / frontend 等版本由该文件锁定；镜像内 ComfyUI 目录
#    不包含 .git，无法 fetch/checkout，故不做版本切换）
RUN python3 -m pip install -q -r requirements.txt || echo "WARN: ComfyUI requirements install failed"

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
# 覆盖基础镜像可能存在的 ENTRYPOINT，确保 RunPod Serverless 执行我们的 start.sh
ENTRYPOINT []
CMD ["bash", "/start.sh"]
