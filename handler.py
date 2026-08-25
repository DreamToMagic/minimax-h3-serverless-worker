"""RunPod Serverless worker for MiniMax H3 (T2V).

Container layout:
- ComfyUI runs locally at 127.0.0.1:8188 (started by /start.sh)
- Models come from the network volume mounted at /runpod-volume
  (see extra_model_paths.yaml)
- On success the generated mp4 is POSTed to our platform webhook:
      POST {WEBHOOK_URL}
      Content-Type: application/octet-stream
      X-Job-Id: <platform job id>
      X-Filename: <mp4 filename>
      X-Signature: hex(HMAC-SHA256(WEBHOOK_SECRET, body))
- Startup self-check fails fast (exit with error) so RunPod schedules a
  fresh worker instead of serving requests on a broken node.
"""
import os
import sys
import time
import json
import hmac
import hashlib
import requests

import runpod

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/workspace/runpod-slim/ComfyUI/output")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "2400"))

MODEL_FILES = {
    "full": "minimax_h3_fl2va_int8_convrot.safetensors",
    "pruned": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
}

REQUIRED_MODELS = [
    ("diffusion_models", "minimax_h3_fl2va_int8_convrot.safetensors"),
    ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("vae", "minimax_h3_video_vae_fp16.safetensors"),
    ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
    ("loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
]


def log(msg):
    print(f"[worker] {msg}", flush=True)


def self_check():
    """Fail fast if the node is broken or the volume is missing models."""
    model_root = "/runpod-volume/models"
    missing = []
    for subdir, name in REQUIRED_MODELS:
        p = os.path.join(model_root, subdir, name)
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        log("FATAL: missing model files:")
        for p in missing:
            log("  - " + p)
        sys.exit(1)

    try:
        import torch
        if not torch.cuda.is_available():
            log("FATAL: CUDA not available")
            sys.exit(1)
        log(f"CUDA ok: {torch.cuda.get_device_name(0)}")
    except Exception as e:  # pragma: no cover
        log(f"FATAL: torch check failed: {e}")
        sys.exit(1)

    for _ in range(30):
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                log("ComfyUI healthy")
                return
        except Exception:
            pass
        time.sleep(2)
    log("FATAL: ComfyUI not healthy")
    sys.exit(1)


def build_graph(prompt, width, height, length, steps, seed, model_file, use_swap=False):
    clip_src = ["3", 0]
    model_src = ["4b", 0]
    extra = {}
    if use_swap:
        # 16/24/32G 显存：UniBlockSwap 把模型块逐块换入换出，牺牲速度换能跑
        extra["3b"] = {"class_type": "UniBlockSwapTE", "inputs": {
            "clip": ["3", 0], "num_blocks": -1}}
        extra["4c"] = {"class_type": "UniBlockSwap", "inputs": {
            "model": ["4b", 0], "num_blocks": -1}}
        clip_src = ["3b", 0]
        model_src = ["4c", 0]
    graph = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "4": {"class_type": "UNETLoader", "inputs": {
            "unet_name": model_file, "weight_dtype": "default"}},
        "4b": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["4", 0],
            "lora_name": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "strength_model": 1.0}},
        "5": {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": {
            "clip": clip_src, "video_vae": ["1", 0], "audio_vae": ["2", 0],
            "prompt": prompt, "width": width, "height": height, "length": length,
            "audio_mode": "native", "audio_denoise_strength": 1.0,
            "add_source_as_reference": True, "prompt_primary_audio_ordinal": 0,
            "strict_prompt_tags": True, "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s", "task_type": "T2VA"}},
        "6": {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": model_src, "av_latent": ["5", 1], "steps": steps,
            "shift_video": 12.0, "shift_audio": 3.0,
            "sampler_name": "dual_clock_euler", "scheduler": "native_flow"}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": ["5", 0]}},
        "9": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["7", 0], "guider": ["8", 0], "sampler": ["6", 1],
            "sigmas": ["6", 2], "latent_image": ["5", 1]}},
        "10": {"class_type": "MiniMaxH3AVDecodeT8", "inputs": {
            "av_latent": ["9", 0], "video_vae": ["1", 0], "audio_vae": ["2", 0]}},
        "11": {"class_type": "VHS_VideoCombine", "inputs": {
            "frame_rate": 24, "loop_count": 0, "filename_prefix": "h3_serverless",
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
            "save_metadata": True, "trim_to_audio": False, "pingpong": False,
            "save_output": True, "images": ["10", 0], "audio": ["10", 1]}},
    }
    graph.update(extra)
    return graph


def submit_prompt(graph):
    client_id = "runpod-h3-" + str(int(time.time()))
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": graph, "client_id": client_id}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "prompt_id" not in data:
        raise RuntimeError("ComfyUI /prompt returned no prompt_id: " + json.dumps(data)[:500])
    return data["prompt_id"]


def collect_outputs(item):
    """Return list of absolute output file paths from a ComfyUI history item."""
    paths = []
    for node in (item.get("outputs") or {}).values():
        for key in ("videos", "images", "gifs"):
            for f in node.get(key) or []:
                filename = f.get("filename")
                if not filename:
                    continue
                subfolder = f.get("subfolder") or ""
                p = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename)
                if os.path.isfile(p):
                    paths.append(p)
    return paths


def poll(prompt_id, timeout=GENERATION_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
            data = r.json()
        except Exception as e:
            log(f"history poll error: {e}")
            time.sleep(5)
            continue
        item = data.get(prompt_id)
        if item and item.get("status", {}).get("completed"):
            paths = collect_outputs(item)
            if not paths:
                raise RuntimeError("ComfyUI completed but no output files found")
            return paths
        if item and item.get("status", {}).get("status_str") == "error":
            raise RuntimeError("ComfyUI generation error: " +
                               json.dumps(item.get("messages", []))[:800])
        time.sleep(5)
    raise TimeoutError("generation timeout")


def upload_to_platform(job_id, filepath, filename):
    """Upload the generated video to our platform webhook.

    Returns True on success. Tries 3 times; if the platform is unreachable the
    job will be reported as failed so the platform can refund the user.
    """
    if not WEBHOOK_URL:
        log("WEBHOOK_URL not configured, skipping upload")
        return False
    with open(filepath, "rb") as f:
        body = f.read()
    signature = hmac.new(WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Job-Id": job_id,
        "X-Filename": filename,
        "X-Signature": signature,
    }
    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.post(WEBHOOK_URL, data=body, headers=headers,
                              timeout=900)
            if r.status_code in (200, 201):
                log(f"uploaded {filename} ({len(body)} bytes) on attempt {attempt}")
                return True
            last_err = f"http {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
        log(f"upload attempt {attempt} failed: {last_err}")
        time.sleep(5)
    raise RuntimeError("webhook upload failed: " + str(last_err))


def get_vram_gb():
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        return round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        return 0


def fit_resolution(w, h, max_area):
    """宽高保持 32 的倍数，缩到不超过 max_area。"""
    if w * h <= max_area:
        return w, h
    s = (max_area / (w * h)) ** 0.5
    nw = max(64, int(w * s / 32) * 32)
    nh = max(64, int(h * s / 32) * 32)
    return nw, nh


def adapt_to_vram(vram_gb, width, height, length, steps):
    """按实际显存自动降级，保证小卡也能跑（T8 README：16GB 需小画布/短片段 + 预检）。"""
    use_swap = False
    if vram_gb <= 0:
        return width, height, length, steps, True, "unknown"
    if vram_gb < 17:
        use_swap = True
        width, height = fit_resolution(width, height, 512 * 288)
        length = min(length, 56)
        steps = min(steps, 8)
        tier = "16G"
    elif vram_gb < 25:
        use_swap = True
        width, height = fit_resolution(width, height, 832 * 480)
        length = min(length, 124)
        steps = min(steps, 20)
        tier = "24G"
    elif vram_gb < 45:
        use_swap = True
        width, height = fit_resolution(width, height, 1024 * 576)
        length = min(length, 124)
        steps = min(steps, 20)
        tier = "32G"
    else:
        tier = "48G+"
    return width, height, length, steps, use_swap, tier


def handler(job):
    job_input = job.get("input", {}) or {}
    job_id = str(job_input.get("job_id") or job.get("id") or "")
    prompt = str(job_input.get("prompt") or "a beautiful 3D anime girl dancing").strip()
    width = int(job_input.get("width") or 640)
    height = int(job_input.get("height") or 640)
    length = int(job_input.get("length") or 22)
    steps = int(job_input.get("steps") or 4)
    seed = int(job_input.get("seed") or 0)
    model = str(job_input.get("model") or "full")
    model_file = MODEL_FILES.get(model, MODEL_FILES["full"])
    # 卷内只保证完整 INT8；pruned 缺失时自动回退，避免坏单
    if not os.path.isfile(os.path.join("/runpod-volume/models", "diffusion_models", model_file)):
        log(f"model file missing ({model_file}), fallback to full INT8")
        model_file = MODEL_FILES["full"]

    vram_gb = get_vram_gb()
    width, height, length, steps, use_swap, tier = adapt_to_vram(vram_gb, width, height, length, steps)
    log(f"job {job_id}: vram={vram_gb}G tier={tier} -> {width}x{height} len={length} steps={steps} swap={use_swap} model={model}")
    graph = build_graph(prompt, width, height, length, steps, seed, model_file, use_swap=use_swap)
    prompt_id = submit_prompt(graph)
    log(f"job {job_id}: submitted prompt_id={prompt_id}")
    paths = poll(prompt_id)

    uploaded = []
    for p in paths:
        filename = os.path.basename(p)
        if job_id and WEBHOOK_URL:
            upload_to_platform(job_id, p, filename)
            uploaded.append(filename)
        else:
            log(f"no webhook configured, keeping file local: {filename}")

    return {
        "status": "COMPLETED",
        "prompt_id": prompt_id,
        "files": [os.path.basename(p) for p in paths],
        "uploaded": uploaded,
        "tier": tier,
        "vram_gb": vram_gb,
        "actual": {"width": width, "height": height, "length": length, "steps": steps},
    }


if __name__ == "__main__":
    self_check()
    log("starting runpod serverless handler")
    runpod.serverless.start({"handler": handler})
