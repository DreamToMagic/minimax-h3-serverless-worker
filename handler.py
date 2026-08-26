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
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/opt/comfyui-baked/output")
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", "/opt/comfyui-baked/input")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
PROGRESS_URL = os.environ.get("PROGRESS_URL", "")
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
    """Check models/CUDA/ComfyUI; called lazily on the first job.

    If the node is broken we kill the container (os._exit) so RunPod marks the
    worker unhealthy and replaces it. Raising an exception would only fail the
    job and leave the bad worker in the pool to hurt the next user.
    """
    model_root = "/runpod-volume/models"
    missing = []
    for subdir, name in REQUIRED_MODELS:
        p = os.path.join(model_root, subdir, name)
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        log("FATAL: missing model files: " + "; ".join(missing))
        os._exit(1)

    try:
        import torch
        if not torch.cuda.is_available():
            log("FATAL: CUDA not available, killing worker for replacement")
            os._exit(1)
        log(f"CUDA ok: {torch.cuda.get_device_name(0)}")
    except SystemExit:
        raise
    except Exception as e:  # pragma: no cover
        log(f"FATAL: torch check failed: {e}, killing worker for replacement")
        os._exit(1)

    for i in range(240):
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                log(f"ComfyUI healthy after {i} checks")
                return
        except Exception:
            pass
        if i % 15 == 14:
            log(f"ComfyUI not ready yet ({i + 1} checks)")
        time.sleep(2)
    log("FATAL: ComfyUI did not become healthy, killing worker for replacement")
    os._exit(1)


AUDIO_STYLE_PROMPTS = {
    "ambient": "背景音乐：轻柔环境音，自然氛围，无强烈旋律。",
    "upbeat": "背景音乐：轻快活泼的电子游戏 BGM，节奏明显，旋律清晰。",
    "electronic": "背景音乐：电子/Synthwave风格，带有合成器旋律和强烈节拍。",
    "cinematic": "背景音乐：电影感配乐，宏大、有氛围感。",
}


def build_graph(prompt, width, height, length, steps, seed, model_file,
                use_swap=False, mode="t2v", refs=None, audio_mode="auto", audio_prompt=""):
    refs = refs or {}
    images = refs.get("images") or []
    videos = refs.get("videos") or []
    audios = refs.get("audios") or []

    # 音频风格并入提示词（与原 H3 工作台一致）
    audio_description = ""
    if audio_mode == "custom":
        audio_description = (audio_prompt or "").strip()
    elif audio_mode == "none":
        audio_description = "No music, only ambient sound. 无音乐，仅环境音。"
    elif AUDIO_STYLE_PROMPTS.get(audio_mode):
        audio_description = AUDIO_STYLE_PROMPTS[audio_mode]
    final_prompt = (prompt + "\n" + audio_description).strip() if audio_description else prompt
    prompt_primary_audio_ordinal = 1 if audio_description else 0

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

    cond = {
        "clip": clip_src, "video_vae": ["1", 0], "audio_vae": ["2", 0],
        "prompt": final_prompt, "width": width, "height": height, "length": length,
        "audio_mode": "native", "audio_denoise_strength": 1.0,
        "add_source_as_reference": True,
        "prompt_primary_audio_ordinal": prompt_primary_audio_ordinal,
        "strict_prompt_tags": True, "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
    }

    if mode == "i2v":
        if not images:
            raise RuntimeError("i2v 需要一张参考图")
        extra["12"] = {"class_type": "LoadImage", "inputs": {"image": images[0]}}
        cond["task_type"] = "I2VA"
        cond["first_frame"] = ["12", 0]
    elif mode == "fl2v":
        if len(images) < 2:
            raise RuntimeError("fl2v 需要首帧图和尾帧图")
        extra["12"] = {"class_type": "LoadImage", "inputs": {"image": images[0]}}
        extra["13"] = {"class_type": "LoadImage", "inputs": {"image": images[1]}}
        cond["task_type"] = "FL2VA"
        cond["first_frame"] = ["12", 0]
        cond["last_frame"] = ["13", 0]
    elif mode == "ref2v":
        if not videos:
            raise RuntimeError("ref2v 需要一段参考视频")
        extra["12"] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": videos[0], "force_rate": 0,
            "custom_width": width, "custom_height": height,
            "frame_load_cap": 0, "skip_first_frames": 0,
            "select_every_nth": 1, "format": "None"}}
        cond["task_type"] = "Ref2VA"
        cond["ref_videos.ref_video_0"] = ["12", 0]
    elif mode == "audio_ref":
        if not audios:
            raise RuntimeError("audio_ref 需要一段参考音频")
        extra["12"] = {"class_type": "LoadAudio", "inputs": {"audio": audios[0]}}
        cond["task_type"] = "T2VA"
        cond["ref_audios.ref_audio_0"] = ["12", 0]
    elif mode == "multi_ref":
        if not (images or videos or audios):
            raise RuntimeError("multi_ref 至少需要一个参考素材")
        cond["task_type"] = "Ref2VA"
        for i, name in enumerate(images[:9]):
            nid = str(20 + i)
            extra[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond[f"ref_images.ref_image_{i}"] = [nid, 0]
        for i, name in enumerate(videos[:3]):
            nid = str(30 + i)
            extra[nid] = {"class_type": "VHS_LoadVideo", "inputs": {
                "video": name, "force_rate": 0,
                "custom_width": width, "custom_height": height,
                "frame_load_cap": 0, "skip_first_frames": 0,
                "select_every_nth": 1, "format": "None"}}
            cond[f"ref_videos.ref_video_{i}"] = [nid, 0]
        for i, name in enumerate(audios[:3]):
            nid = str(40 + i)
            extra[nid] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            cond[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    else:
        cond["task_type"] = "T2VA"

    graph["5"] = {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": cond}
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


def poll(prompt_id, job_id=None, timeout=GENERATION_TIMEOUT):
    deadline = time.time() + timeout
    start = time.time()
    last_report = 0
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
            data = r.json()
        except Exception as e:
            log(f"history poll error: {e}")
            time.sleep(5)
            continue
        item = data.get(prompt_id)
        elapsed = int(time.time() - start)
        if job_id and elapsed - last_report >= 30:
            last_report = elapsed
            report_progress(job_id, "采样生成中", seconds=elapsed)
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


def report_progress(job_id, stage, percent=None, seconds=0):
    """把阶段进度回传平台（前端任务卡显示）。"""
    if not PROGRESS_URL or not job_id:
        return
    try:
        sig = hmac.new(WEBHOOK_SECRET.encode("utf-8"), job_id.encode("utf-8"), hashlib.sha256).hexdigest()
        requests.post(PROGRESS_URL, json={
            "job_id": job_id, "stage": stage, "percent": percent, "seconds": seconds
        }, headers={"X-Signature": sig}, timeout=10)
    except Exception as e:
        log(f"progress report failed: {e}")


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
    """小显存只开 UniBlockSwap 保底，绝不修改用户设定的分辨率/时长/步数。

    若用户参数超出显存能力，ComfyUI 会 OOM 失败，平台按失败退款；
    用户明确要求：不自动降级参数。
    """
    use_swap = vram_gb > 0 and vram_gb < 45
    if vram_gb <= 0:
        tier = "unknown"
    elif vram_gb < 17:
        tier = "16G"
    elif vram_gb < 25:
        tier = "24G"
    elif vram_gb < 45:
        tier = "32G"
    else:
        tier = "48G+"
    return width, height, length, steps, use_swap, tier


_STARTUP_CHECKED = False


def download_refs(refs):
    """下载平台签名 URL 的参考素材到 ComfyUI input 目录。"""
    result = {"images": [], "videos": [], "audios": []}
    if not refs:
        return result
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    for r in refs:
        url = str(r.get("url") or "")
        rtype = str(r.get("type") or "")
        name = os.path.basename(str(r.get("name") or f"ref-{int(time.time())}"))
        safe = "".join(c for c in name if c.isalnum() or c in "._-")[:160] or "ref.bin"
        dest = os.path.join(COMFY_INPUT_DIR, safe)
        log(f"downloading {rtype} ref: {safe}")
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        if rtype == "image":
            result["images"].append(safe)
        elif rtype == "video":
            result["videos"].append(safe)
        elif rtype == "audio":
            result["audios"].append(safe)
    return result


def handler(job):
    global _STARTUP_CHECKED
    if not _STARTUP_CHECKED:
        log("first job received: running startup self-check")
        self_check()
        _STARTUP_CHECKED = True

    job_input = job.get("input", {}) or {}
    job_id = str(job_input.get("job_id") or job.get("id") or "")
    prompt = str(job_input.get("prompt") or "a beautiful 3D anime girl dancing").strip()
    width = int(job_input.get("width") or 640)
    height = int(job_input.get("height") or 640)
    length = int(job_input.get("length") or 22)
    steps = int(job_input.get("steps") or 4)
    seed = int(job_input.get("seed") or 0)
    model = str(job_input.get("model") or "pruned")
    mode = str(job_input.get("mode") or "t2v")
    audio_mode = str(job_input.get("audio_mode") or "auto")
    audio_prompt = str(job_input.get("audio_prompt") or "")
    raw_refs = job_input.get("refs") or []
    refs_requested = bool(raw_refs)
    refs = download_refs(raw_refs)
    model_file = MODEL_FILES.get(model, MODEL_FILES["full"])
    # 卷内只保证完整 INT8；pruned 缺失时自动回退，避免坏单
    if not os.path.isfile(os.path.join("/runpod-volume/models", "diffusion_models", model_file)):
        log(f"model file missing ({model_file}), fallback to full INT8")
        model_file = MODEL_FILES["full"]

    vram_gb = get_vram_gb()
    width, height, length, steps, use_swap, tier = adapt_to_vram(vram_gb, width, height, length, steps)
    log(f"job {job_id}: vram={vram_gb}G tier={tier} -> {width}x{height} len={length} steps={steps} swap={use_swap} model={model}")
    report_progress(job_id, "环境就绪，准备生成", seconds=0)
    if refs_requested:
        report_progress(job_id, "下载参考素材", seconds=0)
    graph = build_graph(prompt, width, height, length, steps, seed, model_file,
                        use_swap=use_swap, mode=mode, refs=refs,
                        audio_mode=audio_mode, audio_prompt=audio_prompt)
    report_progress(job_id, "提交 ComfyUI", seconds=0)
    prompt_id = submit_prompt(graph)
    log(f"job {job_id}: submitted prompt_id={prompt_id}")
    paths = poll(prompt_id, job_id=job_id)

    uploaded = []
    for p in paths:
        filename = os.path.basename(p)
        report_progress(job_id, "上传视频到平台", seconds=0)
        if job_id and WEBHOOK_URL:
            upload_to_platform(job_id, p, filename)
            uploaded.append(filename)
        else:
            log(f"no webhook configured, keeping file local: {filename}")
    report_progress(job_id, "完成", percent=100, seconds=0)

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
    log("starting runpod serverless handler (listening immediately)")
    runpod.serverless.start({"handler": handler})
