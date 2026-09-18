from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import wave
from typing import Any, Optional

import numpy as np
import torch

from .mfg import _find_ffmpeg, _resolve_video_output_path, _nvidia_smi_gpus
from .progress import ConsoleProgress, throw_if_interrupted


class VideoCombineError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return repr(value)


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _prompt_node(prompt: Any, node_id: Any) -> Optional[dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    node = prompt.get(str(node_id))
    if node is None:
        node = prompt.get(node_id)
    return node if isinstance(node, dict) else None


def _walk_upstream(prompt: Any, unique_id: Any, *, max_nodes: int = 512) -> list[tuple[str, dict[str, Any], int]]:
    """Breadth-first walk from this Video Combine's IMAGE input upstream.

    The closest nodes are returned first so the metadata reflects the sampler/model
    that actually feeds this output rather than an unrelated branch elsewhere in
    the workflow.
    """
    current = _prompt_node(prompt, unique_id)
    if current is None:
        return []
    inputs = current.get("inputs") if isinstance(current.get("inputs"), dict) else {}
    start = inputs.get("images")
    if not _is_link(start):
        return []

    queue_items: list[tuple[str, int]] = [(str(start[0]), 0)]
    seen: set[str] = set()
    out: list[tuple[str, dict[str, Any], int]] = []
    while queue_items and len(seen) < int(max_nodes):
        node_id, depth = queue_items.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        node = _prompt_node(prompt, node_id)
        if node is None:
            continue
        out.append((node_id, node, depth))
        node_inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        for value in node_inputs.values():
            if _is_link(value):
                queue_items.append((str(value[0]), depth + 1))
            elif isinstance(value, list):
                for item in value:
                    if _is_link(item):
                        queue_items.append((str(item[0]), depth + 1))
    return out


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _first_scalar(inputs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in inputs:
            value = _scalar(inputs.get(key))
            if value not in (None, ""):
                return value
    return None


def _resolve_linked_scalar(prompt: Any, value: Any, keys: tuple[str, ...]) -> Any:
    if not _is_link(value):
        return _scalar(value)
    node = _prompt_node(prompt, value[0])
    if node is None:
        return None
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    return _first_scalar(inputs, keys)


def _auto_generation_metadata(prompt: Any, unique_id: Any) -> dict[str, Any]:
    nodes = _walk_upstream(prompt, unique_id)
    generation: dict[str, Any] = {
        "seed": None,
        "sampler": None,
        "scheduler": None,
        "model": None,
        "steps": None,
        "cfg": None,
        "denoise": None,
        "source": "auto_graph",
        "source_nodes": {},
    }

    # Collect fields independently. Advanced ComfyUI workflows often split
    # noise/seed, sampler selection and sigma scheduling across separate nodes.
    for node_id, node, _depth in nodes:
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        class_type = str(node.get("class_type", ""))
        low_class = class_type.lower()

        if generation["seed"] is None:
            value = _first_scalar(inputs, ("seed", "noise_seed", "random_seed"))
            if value is not None:
                generation["seed"] = value
                generation["source_nodes"]["seed"] = {"id": node_id, "class_type": class_type}

        if generation["sampler"] is None:
            value = _first_scalar(inputs, ("sampler_name",))
            if value is None and ("ksampler" in low_class or "sampler" in low_class) and "sampler" in inputs:
                value = _resolve_linked_scalar(
                    prompt, inputs.get("sampler"), ("sampler_name", "sampler", "name")
                )
            if value is not None:
                generation["sampler"] = value
                generation["source_nodes"]["sampler"] = {"id": node_id, "class_type": class_type}

        if generation["scheduler"] is None:
            value = _first_scalar(inputs, ("scheduler", "scheduler_name"))
            if value is None and "sigmas" in inputs:
                value = _resolve_linked_scalar(
                    prompt, inputs.get("sigmas"), ("scheduler", "scheduler_name", "name")
                )
            if value is not None:
                generation["scheduler"] = value
                generation["source_nodes"]["scheduler"] = {"id": node_id, "class_type": class_type}

        if generation["steps"] is None:
            value = _first_scalar(inputs, ("steps",))
            if value is not None:
                generation["steps"] = value
        if generation["cfg"] is None:
            value = _first_scalar(inputs, ("cfg", "cfg_scale"))
            if value is not None:
                generation["cfg"] = value
        if generation["denoise"] is None:
            value = _first_scalar(inputs, ("denoise",))
            if value is not None:
                generation["denoise"] = value

    # Find the nearest model/checkpoint loader. Prefer explicit filename-ish
    # fields and avoid serializing MODEL links themselves.
    model_keys = (
        "ckpt_name",
        "checkpoint",
        "checkpoint_name",
        "unet_name",
        "model_name",
        "diffusion_model",
    )
    for node_id, node, _depth in nodes:
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        model_name = _first_scalar(inputs, model_keys)
        if model_name is None:
            continue
        generation["model"] = model_name
        generation["source_nodes"]["model"] = {
            "id": node_id,
            "class_type": str(node.get("class_type", "")),
        }
        break

    # A direct sampler->model edge is a useful second chance for compact graphs.
    if generation["model"] is None:
        for _node_id, node, _depth in nodes:
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            if "model" not in inputs or not _is_link(inputs.get("model")):
                continue
            linked = _prompt_node(prompt, inputs["model"][0])
            if linked is None:
                continue
            linked_inputs = linked.get("inputs") if isinstance(linked.get("inputs"), dict) else {}
            model_name = _first_scalar(linked_inputs, model_keys)
            if model_name is not None:
                generation["model"] = model_name
                generation["source_nodes"]["model"] = {
                    "id": str(inputs["model"][0]),
                    "class_type": str(linked.get("class_type", "")),
                }
                break

    return generation


def _generation_metadata(
    prompt: Any,
    unique_id: Any,
    *,
    generation_seed: int = -1,
    generation_sampler: str = "",
    generation_scheduler: str = "",
    generation_model: str = "",
) -> dict[str, Any]:
    generation = _auto_generation_metadata(prompt, unique_id)
    overrides: dict[str, Any] = {}
    if int(generation_seed) >= 0:
        overrides["seed"] = int(generation_seed)
    if str(generation_sampler).strip():
        overrides["sampler"] = str(generation_sampler).strip()
    if str(generation_scheduler).strip():
        overrides["scheduler"] = str(generation_scheduler).strip()
    if str(generation_model).strip():
        overrides["model"] = str(generation_model).strip()
    if overrides:
        generation.update(overrides)
        generation["source"] = "auto_graph_with_explicit_overrides"
        generation["explicit_overrides"] = sorted(overrides)
    return generation


def _metadata_payload(
    prompt: Any,
    extra_pnginfo: Any,
    *,
    generation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    extra = extra_pnginfo if isinstance(extra_pnginfo, dict) else {}
    return {
        "aetherscale": {"version": "0.9.2", "node": "AetherScaleVideoCombine"},
        "generation": _json_safe(generation or {}),
        "prompt": _json_safe(prompt),
        "workflow": _json_safe(extra.get("workflow")) if isinstance(extra, dict) else None,
        "extra_pnginfo": _json_safe(extra),
    }


def _ffmetadata_escape(value: str) -> str:
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace(";", "\\;")
    value = value.replace("#", "\\#")
    value = value.replace("=", "\\=")
    value = value.replace("\n", "\\\n")
    return value


def _write_temp_ffmetadata(payload: dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="aetherscale_metadata_", suffix=".txt")
    os.close(fd)
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    model = generation.get("model")
    seed = generation.get("seed")
    sampler = generation.get("sampler")
    scheduler = generation.get("scheduler")
    summary_parts = []
    for label, value in (("Model", model), ("Seed", seed), ("Sampler", sampler), ("Scheduler", scheduler)):
        if value not in (None, ""):
            summary_parts.append(f"{label}: {value}")
    summary = " | ".join(summary_parts)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(";FFMETADATA1\n")
        f.write("comment=" + _ffmetadata_escape(compact) + "\n")
        f.write("software=" + _ffmetadata_escape("AetherScale 0.9.2") + "\n")
        if summary:
            f.write("description=" + _ffmetadata_escape(summary) + "\n")
        for key, value in (("model", model), ("seed", seed), ("sampler", sampler), ("scheduler", scheduler)):
            if value not in (None, ""):
                f.write(key + "=" + _ffmetadata_escape(str(value)) + "\n")
    return path


def _sidecar_metadata_path(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_suffix(".metadata.json"))


def _write_sidecar_metadata(video_path: str, payload: dict[str, Any], stats: dict[str, Any]) -> str:
    path = _sidecar_metadata_path(video_path)
    doc = dict(payload)
    doc["output"] = {
        "video_path": video_path,
        "stats": _json_safe(stats),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def _preview_descriptor(output_path: str, display_name: str, save_output: bool, container: str) -> dict[str, Any]:
    display = str(display_name).replace("\\", "/")
    if "/" in display:
        subfolder, filename = display.rsplit("/", 1)
    else:
        subfolder, filename = "", display
    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": "output" if save_output else "temp",
        "format": f"video/{container}",
        "fullpath": output_path,
    }


def _silent_copy_path(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_name(p.stem + "-silent" + p.suffix))


def _make_silent_copy(ffmpeg: str, video_path: str) -> str:
    silent_path = _silent_copy_path(video_path)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        silent_path,
    ]
    cp = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    if cp.returncode != 0:
        detail = cp.stderr.decode("utf-8", errors="replace")[-3000:]
        raise VideoCombineError(f"Failed to create silent copy:\n{detail}")
    return silent_path


def _audio_to_wav(audio: Any) -> Optional[str]:
    """Materialize a ComfyUI AUDIO object to a temporary PCM16 WAV.

    The implementation uses only the stdlib so AetherScale does not depend on
    torchaudio/soundfile. ComfyUI AUDIO is expected to expose `waveform` and
    `sample_rate`.
    """
    if audio is None:
        return None
    if not isinstance(audio, dict):
        raise VideoCombineError("AUDIO must be a ComfyUI audio dictionary.")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
        raise VideoCombineError("AUDIO is missing waveform/sample_rate.")

    x = waveform.detach()
    # Standard ComfyUI layout is [B,C,S]. Use the first batch item.
    while x.ndim > 2:
        x = x[0]
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise VideoCombineError(f"Unsupported AUDIO waveform shape: {tuple(waveform.shape)}")
    if x.device.type != "cpu":
        x = x.to("cpu", non_blocking=False)
    x = x.to(torch.float32).clamp(-1.0, 1.0)
    pcm = (x.transpose(0, 1).contiguous() * 32767.0).round().to(torch.int16)

    fd, path = tempfile.mkstemp(prefix="aetherscale_audio_", suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(pcm.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(memoryview(pcm.numpy()))
    return path


def _parse_audio_bitrate_kbps(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(32, int(value))
    text = str(value or "192").strip().lower()
    m = __import__("re").search(r"(\d+)", text)
    return max(32, int(m.group(1))) if m else 192


_PRORES_PROFILES = {
    "prores_proxy": 0,
    "prores_lt": 1,
    "prores_standard": 2,
    "prores_hq": 3,
    "prores_4444": 4,
    "prores_4444_xq": 5,
}


def _is_prores(codec: str) -> bool:
    return str(codec) in _PRORES_PROFILES


def _prores_output_pix_fmt(codec: str, has_alpha: bool) -> str:
    profile = _PRORES_PROFILES[str(codec)]
    if profile >= 4:
        return "yuva444p10le" if has_alpha else "yuv444p10le"
    return "yuv422p10le"


def _pack_high_depth_chunk(chunk: torch.Tensor, channels: int) -> tuple[str, Any]:
    """Pack normalized BHWC into little-endian 16-bit RGB(A) for ProRes input."""
    x = chunk[..., :channels].detach()
    if x.device.type != "cpu":
        x = x.to("cpu", non_blocking=False)
    frames: list[np.ndarray] = []
    for frame in x:
        if frame.is_floating_point():
            arr = frame.float().numpy()
            packed = np.clip(arr * 65535.0 + 0.5, 0.0, 65535.0).astype("<u2")
        elif frame.dtype == torch.uint8:
            packed = (frame.numpy().astype(np.uint16) * np.uint16(257)).astype("<u2", copy=False)
        else:
            arr = frame.float().numpy()
            packed = np.clip(arr, 0.0, 65535.0).astype("<u2")
        frames.append(np.ascontiguousarray(packed))
    return "numpy_rgb16_frames", frames


def _quantize_chunk(chunk: torch.Tensor, *, input_bit_depth: int = 8, channels: int = 3) -> tuple[str, Any]:
    """Pack a BHWC chunk with an adaptive fast path.

    CPU float32 is unusually fast through NumPy's per-frame vectorized path,
    while CPU float16 and CUDA tensors are substantially faster when quantized
    by Torch in a batch.  `auto` chooses the appropriate path rather than
    forcing one implementation onto every dtype/device.
    """
    if int(input_bit_depth) > 8:
        return _pack_high_depth_chunk(chunk, channels=max(3, int(channels)))
    x = chunk[..., :3].detach()
    if x.device.type == "cpu" and x.dtype == torch.float32:
        frames: list[np.ndarray] = []
        for frame in x:
            arr = frame.numpy()
            packed = np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            frames.append(np.ascontiguousarray(packed))
        return "numpy_float32_frames", frames

    if x.is_floating_point():
        # For float16 / CUDA paths quantize before any device transfer.
        x = (x.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    elif x.dtype != torch.uint8:
        x = x.to(torch.uint8)
    if x.device.type != "cpu":
        x = x.to("cpu", non_blocking=False)
    return "torch_chunk", x.contiguous()


def video_gpu_choices() -> tuple[list[str], str]:
    choices = ["auto"]
    best = "auto"
    best_key = (-1, -1)
    for gpu in _nvidia_smi_gpus():
        idx = int(gpu.get("index", 0))
        name = str(gpu.get("name", "NVIDIA GPU"))
        choice = f"gpu_{idx} | {name}"
        choices.append(choice)
        import re as _re
        m = _re.search(r"RTX\s+(\d{2})", name.upper())
        gen = int(m.group(1)) if m else 0
        key = (gen, int(gpu.get("memory_mb", 0)))
        if key > best_key:
            best_key = key
            best = choice
    return choices, best


def _parse_nvenc_gpu(value: str) -> int | None:
    value = str(value or "auto").strip()
    if value == "auto":
        return None
    import re as _re
    m = _re.match(r"gpu_(\d+)", value)
    if m:
        return int(m.group(1))
    if value.isdigit():
        return int(value)
    return None


def _selected_physical_gpu(value: str) -> dict[str, Any] | None:
    idx = _parse_nvenc_gpu(value)
    if idx is None:
        return None
    for gpu in _nvidia_smi_gpus():
        try:
            if int(gpu.get("index", -1)) == int(idx):
                return dict(gpu)
        except Exception:
            continue
    return {"index": int(idx), "name": f"GPU {idx}", "uuid": "", "pci_bus_id": ""}


def _nvenc_child_env(nvenc_gpu: str) -> tuple[dict[str, str], int | None, dict[str, Any] | None]:
    """Map a physical nvidia-smi choice to FFmpeg's logical CUDA index.

    FFmpeg's NVENC `-gpu N` uses the CUDA-device enumeration visible to the
    child process. ComfyUI can already be running with CUDA_VISIBLE_DEVICES, so
    physical nvidia-smi index 1 can legitimately be FFmpeg logical CUDA index 0.
    Re-isolate the child to the requested physical GPU and always address it as
    logical GPU 0.
    """
    env = os.environ.copy()
    selected = _selected_physical_gpu(nvenc_gpu)
    if selected is None:
        return env, None, None

    uuid = str(selected.get("uuid") or "").strip()
    physical_index = int(selected.get("index", 0))
    selector = uuid if uuid else str(physical_index)
    env["CUDA_VISIBLE_DEVICES"] = selector
    # After CUDA_VISIBLE_DEVICES isolation, the selected physical adapter is the
    # first/only logical CUDA adapter in the FFmpeg child process.
    return env, 0, selected


def _probe_nvenc(
    ffmpeg: str,
    encoder: str,
    *,
    env: dict[str, str],
    logical_gpu: int | None,
    width: int,
    height: int,
    pixel_format: str,
) -> tuple[bool, str]:
    """Open one NVENC session at the real output geometry before streaming.

    Older builds used 64x64 here. Current NVENC drivers can reject that size as
    below the codec minimum, producing a false-negative preflight even though
    the actual video resolution is perfectly valid.
    """
    probe_w = max(1, int(width))
    probe_h = max(1, int(height))
    if str(pixel_format) in {"yuv420p", "yuv422p", "yuv422p10le"}:
        probe_w += probe_w & 1
        probe_h += probe_h & 1
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c=black:s={probe_w}x{probe_h}:r=1",
        "-frames:v", "1",
        "-c:v", str(encoder),
        "-pix_fmt", str(pixel_format),
    ]
    if logical_gpu is not None:
        cmd += ["-gpu", str(int(logical_gpu))]
    cmd += ["-f", "null", "-"]
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = cp.stderr.decode("utf-8", errors="replace")[-2500:]
    return cp.returncode == 0, detail


def _resolve_nvenc_preflight(
    ffmpeg: str,
    requested_encoder: str,
    *,
    nvenc_gpu: str,
    width: int,
    height: int,
    pixel_format: str,
    allow_codec_fallback: bool,
) -> tuple[str, dict[str, str], int | None, dict[str, Any] | None, str, bool]:
    """Resolve an actually usable NVENC encoder/routing before streaming.

    The selected physical GPU is first isolated with CUDA_VISIBLE_DEVICES and
    addressed as logical GPU 0. If that routing is rejected by the FFmpeg
    build, the process-visible automatic route is tried. If the requested
    NVENC codec itself cannot open at the real output geometry, optional codec
    fallback tries other NVIDIA encoders (HEVC/AV1/H.264) instead of wasting a
    long upstream run and failing only at save time.
    """
    selected_env, selected_logical, selected_physical = _nvenc_child_env(str(nvenc_gpu))
    auto_env = os.environ.copy()

    requested = str(requested_encoder)
    candidates = [requested]
    if allow_codec_fallback:
        fallback_order = {
            "h264_nvenc": ["hevc_nvenc", "av1_nvenc"],
            "hevc_nvenc": ["av1_nvenc", "h264_nvenc"],
            "av1_nvenc": ["hevc_nvenc", "h264_nvenc"],
        }.get(requested, [])
        candidates.extend(c for c in fallback_order if c not in candidates)

    attempts: list[str] = []
    for candidate in candidates:
        routes: list[tuple[str, dict[str, str], int | None]] = []
        if selected_physical is not None:
            routes.append(("selected_gpu", selected_env, selected_logical))
        routes.append(("auto_gpu", auto_env, None))

        for route_name, env, logical_gpu in routes:
            ok, detail = _probe_nvenc(
                ffmpeg,
                candidate,
                env=env,
                logical_gpu=logical_gpu,
                width=width,
                height=height,
                pixel_format=pixel_format,
            )
            compact = " ".join(line.strip() for line in detail.splitlines() if line.strip())[-900:]
            attempts.append(f"{candidate}/{route_name}: {'ok' if ok else compact or 'failed'}")
            if ok:
                route_fallback = selected_physical is not None and route_name != "selected_gpu"
                fallback_used = candidate != requested or route_fallback
                return (
                    candidate,
                    env,
                    logical_gpu,
                    selected_physical,
                    " | ".join(attempts),
                    fallback_used,
                )

    selected_name = (
        str(selected_physical.get("name") or nvenc_gpu)
        if selected_physical is not None
        else str(nvenc_gpu)
    )
    raise VideoCombineError(
        "NVENC preflight failed before video streaming. "
        f"Requested codec: {requested}; requested physical GPU: {selected_name}; "
        f"geometry: {int(width)}x{int(height)} {pixel_format}. "
        + " | ".join(attempts)
    )


def _chunk_frames_for_target(height: int, width: int, target_mb: int, bytes_per_pixel: int = 3) -> int:
    frame_bytes = max(1, int(height) * int(width) * max(1, int(bytes_per_pixel)))
    target_bytes = max(1, int(target_mb)) * 1024 * 1024
    return max(1, target_bytes // frame_bytes)


def encode_video_batch(
    images: torch.Tensor,
    *,
    frame_rate: float,
    filename_prefix: str,
    container: str,
    codec: str,
    preset: str,
    nvenc_gpu: str,
    bitrate_mbps: int,
    pixel_format: str,
    audio: Any = None,
    audio_bitrate_kbps: int = 192,
    save_output: bool = True,
    chunk_mb: int = 64,
    pipeline_depth: int = 2,
    save_silent_copy: bool = False,
    save_metadata: bool = True,
    metadata_target: str = "sidecar_json",
    prompt: Any = None,
    extra_pnginfo: Any = None,
    unique_id: Any = None,
    generation_seed: int = -1,
    generation_sampler: str = "",
    generation_scheduler: str = "",
    generation_model: str = "",
    nvenc_codec_fallback: bool = True,
) -> dict[str, Any]:
    """Fast generic IMAGE->video encoder for ComfyUI.

    Key differences from common Python-frame loops:
    - converts several frames per Torch chunk instead of one NumPy conversion
      per frame;
    - uses the buffer protocol directly (`memoryview`) rather than `.tobytes()`;
    - pre-converts the next chunk in a bounded producer thread while FFmpeg is
      consuming the current chunk;
    - NVENC is the default encoder.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise VideoCombineError(f"Expected IMAGE [B,H,W,C], got {getattr(images, 'shape', None)}")
    n, h, w, c = [int(v) for v in images.shape]
    if n <= 0:
        raise VideoCombineError("Video Combine received an empty IMAGE batch.")
    if c < 3:
        raise VideoCombineError(f"Video Combine requires at least 3 channels, got {c}.")
    if float(frame_rate) <= 0:
        raise VideoCombineError("frame_rate must be > 0.")

    ffmpeg = _find_ffmpeg()
    ffmpeg_env = os.environ.copy()
    nvenc_logical_gpu: int | None = None
    nvenc_selected_physical: dict[str, Any] | None = None
    nvenc_probe_detail = ""
    requested_container = str(container).lower()
    if requested_container not in {"mp4", "mkv", "mov"}:
        raise VideoCombineError(f"Unsupported container: {requested_container}")
    requested_encoder = str(codec)
    encoder = requested_encoder
    prores = _is_prores(encoder)
    container = "mov" if prores else requested_container
    output_path, display_name = _resolve_video_output_path(filename_prefix, container)
    if not save_output:
        basename = Path(output_path).name
        try:
            import folder_paths  # type: ignore
            temp_root = Path(folder_paths.get_temp_directory()) / "AetherScale"
            temp_root.mkdir(parents=True, exist_ok=True)
            output_path = str(temp_root / basename)
            display_name = f"AetherScale/{basename}"
        except Exception:
            temp_root = Path(tempfile.gettempdir()) / "aetherscale"
            temp_root.mkdir(parents=True, exist_ok=True)
            output_path = str(temp_root / basename)
            display_name = basename

    audio_path = _audio_to_wav(audio)
    metadata_target = str(metadata_target or "sidecar_json")
    if metadata_target not in {"video_container", "sidecar_json", "both"}:
        metadata_target = "sidecar_json"
    generation = _generation_metadata(
        prompt,
        unique_id,
        generation_seed=int(generation_seed),
        generation_sampler=str(generation_sampler),
        generation_scheduler=str(generation_scheduler),
        generation_model=str(generation_model),
    )
    metadata_payload = _metadata_payload(prompt, extra_pnginfo, generation=generation) if save_metadata else None
    metadata_path = None
    if metadata_payload is not None and metadata_target in {"video_container", "both"}:
        metadata_path = _write_temp_ffmetadata(metadata_payload)
    has_alpha = c >= 4
    input_bit_depth = 16 if prores else 8
    input_channels = 4 if (prores and has_alpha and _PRORES_PROFILES.get(encoder, 0) >= 4) else 3
    input_pix_fmt = "rgba64le" if input_channels == 4 else ("rgb48le" if prores else "rgb24")
    effective_pixel_format = _prores_output_pix_fmt(encoder, has_alpha and input_channels == 4) if prores else str(pixel_format)
    audio_bitrate = _parse_audio_bitrate_kbps(audio_bitrate_kbps)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s:v", f"{w}x{h}",
        "-framerate", f"{float(frame_rate):.8f}",
        "-i", "-",
    ]
    if audio_path:
        cmd += ["-i", audio_path]
    metadata_input_index = None
    if metadata_path:
        metadata_input_index = 2 if audio_path else 1
        cmd += ["-f", "ffmetadata", "-i", metadata_path]
    cmd += ["-map", "0:v:0"]
    if audio_path:
        cmd += ["-map", "1:a:0"]
    if metadata_input_index is not None:
        cmd += ["-map_metadata", str(metadata_input_index)]
    if not prores:
        cmd += ["-c:v", encoder]

    if encoder.endswith("_nvenc"):
        # GPU routing and the actually usable NVENC codec are resolved by the
        # real-geometry preflight below before the FFmpeg process is started.
        pass
    elif encoder == "libx264":
        cpu_preset = {
            "p1": "ultrafast",
            "p2": "superfast",
            "p3": "veryfast",
            "p4": "faster",
            "p5": "fast",
            "p6": "medium",
            "p7": "slow",
        }.get(str(preset), "veryfast")
        cmd += ["-preset", cpu_preset, "-b:v", f"{int(bitrate_mbps)}M"]
    elif prores:
        cmd += ["-c:v", "prores_ks", "-profile:v", str(_PRORES_PROFILES[encoder])]
    else:
        raise VideoCombineError(f"Unsupported codec: {encoder}")

    # yuv420 encoders need even dimensions. Padding is no-op for normal even
    # sizes and cheaper/safer than rejecting an otherwise valid ComfyUI batch.
    if (effective_pixel_format in {"yuv420p", "yuv422p10le"}) and ((w & 1) or (h & 1)):
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    cmd += ["-pix_fmt", effective_pixel_format]

    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", f"{audio_bitrate}k", "-shortest"]
    else:
        cmd += ["-an"]
    if container in {"mp4", "mov"}:
        movflags = "+faststart+use_metadata_tags" if metadata_path else "+faststart"
        cmd += ["-movflags", movflags]
    cmd += [output_path]

    if encoder.endswith("_nvenc"):
        print(
            f"[AetherScale] Video Combine NVENC preflight: {w}x{h} {effective_pixel_format} "
            f"| requested {requested_encoder} on {nvenc_gpu}",
            flush=True,
        )
        (
            encoder,
            ffmpeg_env,
            nvenc_logical_gpu,
            nvenc_selected_physical,
            nvenc_probe_detail,
            nvenc_fallback_used,
        ) = _resolve_nvenc_preflight(
            ffmpeg,
            requested_encoder,
            nvenc_gpu=str(nvenc_gpu),
            width=w,
            height=h,
            pixel_format=effective_pixel_format,
            allow_codec_fallback=bool(nvenc_codec_fallback),
        )
        if encoder != requested_encoder:
            print(
                f"[AetherScale] Video Combine NVENC codec fallback: "
                f"{requested_encoder} -> {encoder} for {w}x{h}",
                flush=True,
            )

        # Insert the resolved encoder and GPU routing into the final command.
        # `-c:v` already exists from the generic non-ProRes command prefix.
        try:
            codec_pos = cmd.index("-c:v")
            cmd[codec_pos + 1] = encoder
        except ValueError:
            cmd += ["-c:v", encoder]
        nvenc_args: list[str] = []
        if nvenc_logical_gpu is not None:
            nvenc_args += ["-gpu", str(int(nvenc_logical_gpu))]
        nvenc_args += [
            "-preset", str(preset),
            "-rc", "vbr",
            "-b:v", f"{int(bitrate_mbps)}M",
            "-maxrate", f"{max(int(bitrate_mbps), 1) * 2}M",
            "-bufsize", f"{max(int(bitrate_mbps), 1) * 4}M",
        ]
        # Output-specific options must appear before the output filename.
        cmd[-1:-1] = nvenc_args
    else:
        nvenc_fallback_used = False

    target_chunk_frames = _chunk_frames_for_target(h, w, int(chunk_mb), bytes_per_pixel=input_channels * (2 if prores else 1))
    depth = max(1, min(int(pipeline_depth), 4))
    q: queue.Queue[Any] = queue.Queue(maxsize=depth)
    stop = threading.Event()
    sentinel = object()

    conversion_seconds = 0.0
    converted_frames = 0

    def put_item(item: Any) -> bool:
        while not stop.is_set():
            throw_if_interrupted()
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        nonlocal conversion_seconds, converted_frames
        try:
            for start in range(0, n, target_chunk_frames):
                throw_if_interrupted()
                if stop.is_set():
                    break
                end = min(n, start + target_chunk_frames)
                t0 = time.perf_counter()
                packing_mode, packed = _quantize_chunk(images[start:end], input_bit_depth=input_bit_depth, channels=input_channels)
                conversion_seconds += time.perf_counter() - t0
                converted_frames += int(end - start)
                if not put_item((start, end, packing_mode, packed)):
                    return
            put_item(sentinel)
        except BaseException as exc:  # forward producer failure to main thread
            put_item(exc)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    start_time = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=ffmpeg_env,
        creationflags=creationflags,
    )
    if proc.stdin is None:
        raise VideoCombineError("Failed to open FFmpeg stdin pipe.")

    producer_thread = threading.Thread(
        target=producer,
        name="AetherScaleVideoPacker",
        daemon=True,
    )
    producer_thread.start()
    written_frames = 0
    write_seconds = 0.0
    packing_modes: set[str] = set()
    console_progress = ConsoleProgress("Video Combine / encode", n, unit="frame")

    try:
        while True:
            throw_if_interrupted()
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    detail = proc.stderr.read() if proc.stderr is not None else b""
                    raise VideoCombineError(
                        "FFmpeg exited while waiting for video frames: "
                        + detail.decode("utf-8", errors="replace")[-5000:]
                    )
                continue
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            _start, end, packing_mode, packed = item
            packing_modes.add(str(packing_mode))
            t0 = time.perf_counter()
            if packing_mode == "torch_chunk":
                # NumPy exposes Torch CPU storage through the buffer protocol.
                # No separate `.tobytes()` allocation is created.
                proc.stdin.write(memoryview(packed.numpy()))
                frames = int(packed.shape[0])
            else:
                frames = len(packed)
                for frame in packed:
                    proc.stdin.write(memoryview(frame))
            write_seconds += time.perf_counter() - t0
            written_frames += frames
            console_progress.update(frames)
            del packed

        throw_if_interrupted()
        proc.stdin.flush()
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[-5000:]
            try:
                os.remove(output_path)
            except OSError:
                pass
            raise VideoCombineError(f"FFmpeg encode failed with exit code {returncode}:\n{detail}")
    except BrokenPipeError as exc:
        console_progress.close(status="failed")
        stop.set()
        detail = b""
        try:
            detail = proc.stderr.read() if proc.stderr is not None else b""
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise VideoCombineError(
            "FFmpeg closed the input pipe early: "
            + detail.decode("utf-8", errors="replace")[-5000:]
        ) from exc
    except BaseException:
        console_progress.close(status="failed")
        stop.set()
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        raise
    finally:
        stop.set()
        producer_thread.join(timeout=2.0)
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass
        if metadata_path:
            try:
                os.remove(metadata_path)
            except OSError:
                pass

    console_progress.close(status="done")
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    stats = {
        "output_path": output_path,
        "display_name": display_name,
        "container": container,
        "requested_container": requested_container,
        "container_auto_forced_to_mov": bool(prores and requested_container != "mov"),
        "codec": encoder,
        "requested_codec": requested_encoder,
        "encoder_resolved": "prores_ks" if prores else encoder,
        "nvenc_codec_fallback_enabled": bool(nvenc_codec_fallback),
        "nvenc_codec_fallback_used": bool(nvenc_fallback_used),
        "prores_profile": _PRORES_PROFILES.get(encoder),
        "preset": str(preset),
        "nvenc_gpu": str(nvenc_gpu),
        "nvenc_ffmpeg_logical_gpu": nvenc_logical_gpu,
        "nvenc_selected_physical": nvenc_selected_physical,
        "nvenc_cuda_visible_devices": ffmpeg_env.get("CUDA_VISIBLE_DEVICES"),
        "nvenc_preflight": "ok" if encoder.endswith("_nvenc") else "not_applicable",
        "nvenc_preflight_detail": nvenc_probe_detail,
        "bitrate_mbps": int(bitrate_mbps),
        "bitrate_ignored_for_prores": bool(prores),
        "pixel_format": effective_pixel_format,
        "input_pixel_format": input_pix_fmt,
        "input_bit_depth": input_bit_depth,
        "audio_bitrate_kbps": int(audio_bitrate),
        "frame_rate": float(frame_rate),
        "frames": int(written_frames),
        "resolution": [w, h],
        "source_dtype": str(images.dtype).replace("torch.", ""),
        "source_device": str(images.device),
        "chunk_mb": int(chunk_mb),
        "frames_per_chunk": int(target_chunk_frames),
        "pipeline_depth": int(depth),
        "producer_consumer_pipeline": True,
        "packing_modes": sorted(packing_modes),
        "numpy_per_frame_loop": "numpy_float32_frames" in packing_modes,
        "uses_tobytes": False,
        "audio_connected": bool(audio is not None),
        "audio_muxed": bool(audio_path is not None),
        "save_silent_copy": bool(save_silent_copy),
        "elapsed_seconds": round(elapsed, 3),
        "encode_pipeline_fps": round(written_frames / elapsed, 3),
        "conversion_seconds_accumulated": round(conversion_seconds, 3),
        "pipe_write_seconds_accumulated": round(write_seconds, 3),
        "output_bytes": int(size_bytes),
        "ffmpeg": ffmpeg,
        "generation_metadata": _json_safe(generation),
        "save_metadata": bool(save_metadata),
        "metadata_target": metadata_target if save_metadata else "disabled",
    }

    silent_copy_path = None
    if audio is not None and save_silent_copy:
        silent_copy_path = _make_silent_copy(ffmpeg, output_path)
    stats["silent_copy_path"] = silent_copy_path

    sidecar_path = None
    if metadata_payload is not None and metadata_target in {"sidecar_json", "both"}:
        sidecar_path = _write_sidecar_metadata(output_path, metadata_payload, stats)
    stats["metadata_sidecar_path"] = sidecar_path
    stats["metadata_embedded"] = bool(metadata_payload is not None and metadata_target in {"video_container", "both"})
    stats["preview"] = _preview_descriptor(output_path, display_name, bool(save_output), container)
    stats["output_files"] = [p for p in [output_path, silent_copy_path, sidecar_path] if p]
    return stats
