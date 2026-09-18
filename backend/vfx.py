from __future__ import annotations

from dataclasses import dataclass, field
import gc
import inspect
import queue
import threading
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from .runtime import RuntimeManager
from .progress import ConsoleProgress, throw_if_interrupted
from .storage import allocate_cpu_tensor, resolve_dtype, sync_file_backed_tensor


QUALITY_MAP: Dict[Tuple[str, str], str] = {
    ("compressed", "low"): "LOW",
    ("compressed", "medium"): "MEDIUM",
    ("compressed", "high"): "HIGH",
    ("compressed", "ultra"): "ULTRA",
    ("high_bitrate", "low"): "HIGHBITRATE_LOW",
    ("high_bitrate", "medium"): "HIGHBITRATE_MEDIUM",
    ("high_bitrate", "high"): "HIGHBITRATE_HIGH",
    ("high_bitrate", "ultra"): "HIGHBITRATE_ULTRA",
    ("denoise", "low"): "DENOISE_LOW",
    ("denoise", "medium"): "DENOISE_MEDIUM",
    ("denoise", "high"): "DENOISE_HIGH",
    ("denoise", "ultra"): "DENOISE_ULTRA",
    ("deblur", "low"): "DEBLUR_LOW",
    ("deblur", "medium"): "DEBLUR_MEDIUM",
    ("deblur", "high"): "DEBLUR_HIGH",
    ("deblur", "ultra"): "DEBLUR_ULTRA",
}

HDR_CLASS_CANDIDATES = (
    "VideoHDR",
    "VideoHdr",
    "RTXVideoHDR",
    "RTXVideoHdr",
    "HDRVideo",
)

HDR_QUALITY_ENUM_CANDIDATES = (
    "QualityLevel",
    "HDRQualityLevel",
)

HDR_QUALITY_NAME_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "low": ("LOW", "QUALITY_LOW"),
    "medium": ("MEDIUM", "QUALITY_MEDIUM"),
    "high": ("HIGH", "QUALITY_HIGH"),
    "ultra": ("ULTRA", "QUALITY_ULTRA"),
}


@dataclass(frozen=True, slots=True)
class VFXConfig:
    effect_type: str
    mode: str
    quality: str
    out_width: int
    out_height: int
    device: int
    extras: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


class _EffectCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._effects: Dict[VFXConfig, Any] = {}

    def get(self, config: VFXConfig) -> Any:
        with self._lock:
            effect = self._effects.get(config)
            if effect is not None:
                return effect
            effect = _build_effect(config)
            self._effects[config] = effect
            return effect

    def evict_except(self, keep: Optional[VFXConfig] = None) -> None:
        with self._lock:
            for key in list(self._effects):
                if keep is not None and key == keep:
                    continue
                effect = self._effects.pop(key)
                try:
                    effect.close()
                except Exception:
                    pass

    def clear(self) -> None:
        self.evict_except(None)

    def size(self) -> int:
        with self._lock:
            return len(self._effects)


_CACHE = _EffectCache()


def _round_dim(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(round(value / multiple)) * multiple)


def resolve_output_size(
    width: int,
    height: int,
    resize_mode: str,
    scale: float,
    target_width: int,
    target_height: int,
    long_edge: int,
    alignment: int,
) -> Tuple[int, int]:
    if resize_mode == "scale":
        out_w = width * scale
        out_h = height * scale
    elif resize_mode == "exact":
        out_w = target_width if target_width > 0 else width
        out_h = target_height if target_height > 0 else height
    elif resize_mode == "long_edge":
        src_long = max(width, height)
        if src_long <= 0:
            raise ValueError("Invalid source dimensions.")
        ratio = long_edge / src_long
        out_w = width * ratio
        out_h = height * ratio
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")

    return _round_dim(out_w, alignment), _round_dim(out_h, alignment)


def _cuda_memory(device: int) -> Tuple[int, int]:
    with torch.cuda.device(device):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except TypeError:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def _release_comfy_models() -> str:
    messages = []
    try:
        import comfy.model_management as mm

        unload = getattr(mm, "unload_all_models", None)
        if callable(unload):
            unload()
            messages.append("unload_all_models")

        soft_empty = getattr(mm, "soft_empty_cache", None)
        if callable(soft_empty):
            try:
                soft_empty()
            except TypeError:
                soft_empty(force=True)
            messages.append("soft_empty_cache")
    except Exception as exc:
        messages.append(f"comfy_release_unavailable:{type(exc).__name__}")

    gc.collect()
    try:
        torch.cuda.empty_cache()
        messages.append("torch_empty_cache")
    except Exception:
        pass
    return ",".join(messages)


def _ensure_runtime_module():
    RuntimeManager.ensure()
    import nvvfx
    return nvvfx


def _find_first_attr(module: Any, names: Iterable[str]) -> Optional[Any]:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return None


def _safe_set_attr(effect: Any, names: Iterable[str], value: Any) -> bool:
    for name in names:
        if hasattr(effect, name):
            try:
                setattr(effect, name, value)
                return True
            except Exception:
                continue
    return False


def _build_video_super_res(config: VFXConfig) -> Any:
    nvvfx = _ensure_runtime_module()
    effect_cls = getattr(nvvfx, "VideoSuperRes", None)
    if effect_cls is None:
        raise RuntimeError("The installed NVIDIA runtime does not expose VideoSuperRes.")

    enum_name = QUALITY_MAP[(config.mode, config.quality)]
    enum_value = getattr(effect_cls.QualityLevel, enum_name)

    effect = effect_cls(quality=enum_value, device=config.device)
    effect.output_width = config.out_width
    effect.output_height = config.out_height
    effect.load()
    return effect


def _build_video_hdr(config: VFXConfig) -> Any:
    nvvfx = _ensure_runtime_module()
    effect_cls = _find_first_attr(nvvfx, HDR_CLASS_CANDIDATES)
    if effect_cls is None:
        available = ", ".join(sorted(name for name in dir(nvvfx) if "hdr" in name.lower()))
        raise RuntimeError(
            "This NVIDIA runtime does not expose an HDR effect compatible with AetherScale. "
            f"HDR-like symbols detected: {available or 'none'}"
        )

    effect = None
    ctor_attempts = [
        {"device": config.device},
        {"gpu_id": config.device},
        {"cuda_device": config.device},
        {},
    ]
    for kwargs in ctor_attempts:
        try:
            effect = effect_cls(**kwargs)
            break
        except TypeError:
            continue
    if effect is None:
        raise RuntimeError(f"Could not construct HDR effect class {effect_cls.__name__}.")

    extras = dict(config.extras)

    enum_container = _find_first_attr(effect_cls, HDR_QUALITY_ENUM_CANDIDATES)
    if enum_container is not None:
        for candidate in HDR_QUALITY_NAME_CANDIDATES.get(config.quality, ()):
            enum_value = getattr(enum_container, candidate, None)
            if enum_value is not None:
                _safe_set_attr(effect, ("quality", "quality_level"), enum_value)
                break

    # Best-effort parameter mapping. Unsupported controls are ignored.
    if "strength" in extras:
        _safe_set_attr(effect, ("strength", "hdr_strength", "effect_strength"), float(extras["strength"]))
    if "saturation" in extras:
        _safe_set_attr(effect, ("saturation", "color_saturation"), float(extras["saturation"]))
    if "contrast" in extras:
        _safe_set_attr(effect, ("contrast", "tonemap_contrast"), float(extras["contrast"]))
    if "highlight_preservation" in extras:
        _safe_set_attr(
            effect,
            ("highlight_preservation", "preserve_highlights", "highlight_strength"),
            float(extras["highlight_preservation"]),
        )
    if "mode_profile" in extras:
        _safe_set_attr(effect, ("profile", "mode", "output_profile"), str(extras["mode_profile"]))

    _safe_set_attr(effect, ("output_width", "width"), config.out_width)
    _safe_set_attr(effect, ("output_height", "height"), config.out_height)

    if hasattr(effect, "load"):
        effect.load()
    return effect


def _build_effect(config: VFXConfig) -> Any:
    if config.effect_type == "video_super_res":
        return _build_video_super_res(config)
    if config.effect_type == "video_hdr":
        return _build_video_hdr(config)
    raise ValueError(f"Unsupported effect_type: {config.effect_type}")


def _prepare_vram(vram_guard: str, min_free_vram_mb: int, device: int) -> Tuple[int, int, str, str]:
    before_free, total_vram = _cuda_memory(device)
    release_reason = "none"
    release_actions = ""

    min_free_bytes = max(0, int(min_free_vram_mb)) * 1024 * 1024
    if vram_guard == "release_models":
        release_reason = "forced"
        release_actions = _release_comfy_models()
    elif vram_guard == "auto" and before_free < min_free_bytes:
        release_reason = f"auto_below_{min_free_vram_mb}MB"
        release_actions = _release_comfy_models()

    after_release_free, _ = _cuda_memory(device)
    return before_free, after_release_free, total_vram, release_reason, release_actions


def _finalize_output_device(out_cpu: torch.Tensor, src_device: torch.device, output_device: str) -> torch.Tensor:
    if output_device == "same_as_input" and src_device.type != "cpu":
        return out_cpu.to(src_device, non_blocking=False)
    return out_cpu


def _append_alpha_if_needed(
    input_images: torch.Tensor,
    out_cpu_rgb: torch.Tensor,
    out_height: int,
    out_width: int,
) -> torch.Tensor:
    if int(input_images.shape[-1]) <= 3:
        return out_cpu_rgb

    alpha_cpu = input_images[..., 3:4].to(device="cpu", dtype=torch.float32)
    alpha_bchw = alpha_cpu.permute(0, 3, 1, 2).contiguous()
    alpha_out = F.interpolate(
        alpha_bchw,
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous().clamp_(0.0, 1.0)
    return torch.cat([out_cpu_rgb, alpha_out], dim=-1)


def _stream_video_super_res(
    images_bhwc: torch.Tensor,
    *,
    config: VFXConfig,
    cache_policy: str,
    cuda_stream_mode: str,
    memory_policy: str,
    output_device: str,
    output_precision: str = "auto",
    output_storage: str = "auto",
    clean_cache: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Stream NVIDIA VSR with GPU/CPU write overlap.

    The v0.8.7 path serialized GPU VSR -> D2H -> mmap write -> FlushViewOfFile
    for every frame. On large disk-spill outputs that made the RTX wait on the
    SATA SSD and looked like CPU-only processing. This path keeps only two
    pinned output frames and lets a writer thread persist frame N while NVIDIA
    VFX computes frame N+1. The full output is still bounded by the storage
    layer and never materialized as an extra RAM copy.
    """
    src_device = images_bhwc.device
    cuda_device = torch.device(f"cuda:{config.device}")
    batch = int(images_bhwc.shape[0])
    channels = 4 if int(images_bhwc.shape[-1]) > 3 else 3
    shape = (batch, config.out_height, config.out_width, channels)
    out_dtype = resolve_dtype(
        requested=output_precision,
        shape=shape,
        input_dtype=images_bhwc.dtype,
    )
    out_cpu, storage = allocate_cpu_tensor(
        shape,
        dtype=out_dtype,
        storage_mode=output_storage,
        prefix="vsr",
        clean_cache=bool(clean_cache),
    )
    label = "Super Resolution" if config.mode in {"high_bitrate", "compressed", "bicubic"} else f"Restoration / {config.mode}"
    console_progress = ConsoleProgress(label, batch, unit="frame")
    pipeline_mode = "cpu_bicubic"
    staging_pinned = False

    try:
        if config.mode == "bicubic":
            for i in range(batch):
                throw_if_interrupted()
                frame = images_bhwc[i, ..., :3].to(dtype=torch.float32)
                bchw = frame.permute(2, 0, 1).unsqueeze(0).contiguous()
                resized = F.interpolate(
                    bchw,
                    size=(config.out_height, config.out_width),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0].permute(1, 2, 0).contiguous().clamp_(0, 1)
                out_cpu[i, ..., :3].copy_(resized.to("cpu", dtype=out_dtype))
                del frame, bchw, resized
                sync_file_backed_tensor(
                    out_cpu,
                    bytes_written=out_cpu[i, ..., :3].numel() * out_cpu.element_size(),
                )
                console_progress.update(1)
        else:
            pipeline_mode = "nvidia_vfx_gpu_overlapped_writer"
            if cache_policy == "single":
                _CACHE.evict_except(config)
            elif cache_policy == "none":
                _CACHE.clear()
            effect = _CACHE.get(config)

            # Two frame-sized staging buffers are enough to overlap GPU compute
            # with disk/RAM persistence while keeping pinned memory bounded.
            staging: list[torch.Tensor] = []
            try:
                staging = [
                    torch.empty(
                        (config.out_height, config.out_width, 3),
                        dtype=out_dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    for _ in range(2)
                ]
                staging_pinned = True
            except Exception:
                staging = [
                    torch.empty(
                        (config.out_height, config.out_width, 3),
                        dtype=out_dtype,
                        device="cpu",
                    )
                    for _ in range(2)
                ]
                staging_pinned = False

            free_slots: "queue.Queue[int]" = queue.Queue(maxsize=2)
            for slot_index in range(2):
                free_slots.put(slot_index)
            write_q: "queue.Queue[object]" = queue.Queue(maxsize=2)
            sentinel = object()
            stop = threading.Event()
            writer_errors: list[BaseException] = []

            def _get_free_slot() -> int:
                while True:
                    throw_if_interrupted()
                    if writer_errors:
                        raise writer_errors[0]
                    try:
                        return free_slots.get(timeout=0.05)
                    except queue.Empty:
                        continue

            def _writer() -> None:
                try:
                    while not stop.is_set():
                        try:
                            item = write_q.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if item is sentinel:
                            return
                        slot_index, frame_index, ready_event, keepalive = item  # type: ignore[misc]
                        ready_event.synchronize()
                        if stop.is_set():
                            return
                        out_cpu[frame_index, ..., :3].copy_(staging[slot_index], non_blocking=False)
                        sync_file_backed_tensor(
                            out_cpu,
                            bytes_written=out_cpu[frame_index, ..., :3].numel() * out_cpu.element_size(),
                        )
                        # `keepalive` owns the DLPack view/result until D2H has
                        # completed. Releasing it before event synchronization can
                        # invalidate runtime-owned output storage.
                        del keepalive
                        free_slots.put(slot_index)
                except BaseException as exc:
                    writer_errors.append(exc)
                    stop.set()
                    # Wake the producer if it is waiting for a staging slot.
                    try:
                        free_slots.put_nowait(0)
                    except queue.Full:
                        pass

            writer_thread = threading.Thread(
                target=_writer,
                name="AetherScaleVSRWriter",
                daemon=True,
            )
            writer_thread.start()

            try:
                with torch.cuda.device(cuda_device):
                    stream = (
                        torch.cuda.Stream(device=cuda_device)
                        if cuda_stream_mode == "dedicated"
                        else torch.cuda.current_stream(device=cuda_device)
                    )
                    for i in range(batch):
                        throw_if_interrupted()
                        slot_index = _get_free_slot()
                        if writer_errors:
                            raise writer_errors[0]

                        frame = (
                            images_bhwc[i, ..., :3]
                            .to(cuda_device, dtype=torch.float32, non_blocking=False)
                            .clamp_(0, 1)
                            .permute(2, 0, 1)
                            .contiguous()
                        )
                        result = effect.run(
                            frame,
                            non_blocking=False,
                            stream_ptr=int(stream.cuda_stream),
                        )
                        view = torch.from_dlpack(result.image).permute(1, 2, 0)
                        with torch.cuda.stream(stream):
                            staging[slot_index].copy_(
                                view,
                                non_blocking=bool(staging_pinned),
                            )
                            ready_event = torch.cuda.Event(blocking=False)
                            ready_event.record(stream)

                        # Backpressure is bounded to two staged frames. Disk I/O
                        # can run concurrently with the next effect invocation.
                        while True:
                            throw_if_interrupted()
                            if writer_errors:
                                raise writer_errors[0]
                            try:
                                write_q.put(
                                    (slot_index, i, ready_event, (view, result)),
                                    timeout=0.05,
                                )
                                break
                            except queue.Full:
                                continue
                        del frame
                        if memory_policy == "aggressive" and (i + 1) % 32 == 0:
                            # Emptying CUDA cache every frame globally synchronizes
                            # the allocator and destroys overlap. 32-frame cadence
                            # retains the low-VRAM intent without serializing VSR.
                            torch.cuda.empty_cache()
                        console_progress.update(1)

                # Wait interruptibly for the CPU writer to persist its final jobs.
                while not write_q.empty() or free_slots.qsize() < 2:
                    throw_if_interrupted()
                    if writer_errors:
                        raise writer_errors[0]
                    time.sleep(0.02)
                while True:
                    try:
                        write_q.put(sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        throw_if_interrupted()
                while writer_thread.is_alive():
                    throw_if_interrupted()
                    writer_thread.join(timeout=0.05)
                if writer_errors:
                    raise writer_errors[0]
            except BaseException:
                stop.set()
                try:
                    write_q.put_nowait(sentinel)
                except queue.Full:
                    pass
                writer_thread.join(timeout=1.0)
                raise
            finally:
                stop.set()

        # Alpha is streamed frame-by-frame; never materialize BxHxW alpha temp.
        if channels == 4:
            for i in range(batch):
                throw_if_interrupted()
                a = images_bhwc[i:i + 1, ..., 3:4].to(device="cpu", dtype=torch.float32)
                a = F.interpolate(
                    a.permute(0, 3, 1, 2),
                    size=(config.out_height, config.out_width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)[0]
                out_cpu[i, ..., 3:4].copy_(a.to(dtype=out_dtype))
                sync_file_backed_tensor(
                    out_cpu,
                    bytes_written=out_cpu[i, ..., 3:4].numel() * out_cpu.element_size(),
                )
                del a

        sync_file_backed_tensor(out_cpu, force=True)
        console_progress.close(status="done")
    except BaseException:
        console_progress.close(status="failed")
        raise
    finally:
        if cache_policy == "none":
            _CACHE.clear()

    throw_if_interrupted()
    if output_device == "same_as_input" and src_device.type != "cpu":
        # Explicitly requested: may be huge and can OOM.
        out = out_cpu.to(src_device, non_blocking=False)
    else:
        out = out_cpu
    return out, {
        "output_storage_backend": storage.backend,
        "output_storage_path": storage.path,
        "output_precision": storage.dtype,
        "output_bytes": storage.bytes,
        "output_gib": round(storage.bytes / 1024 ** 3, 3),
        "output_storage_fallback_reason": storage.fallback_reason,
        "system_commit_available_gib_at_allocation": (
            round(storage.commit_available_bytes / 1024 ** 3, 3)
            if storage.commit_available_bytes is not None else None
        ),
        "spill_disk_free_gib_at_allocation": (
            round(storage.disk_free_bytes / 1024 ** 3, 3)
            if storage.disk_free_bytes is not None else None
        ),
        "pipeline_mode": pipeline_mode,
        "gpu_cpu_overlap": bool(config.mode != "bicubic"),
        "pinned_staging": bool(staging_pinned),
        "staging_frames": 2 if config.mode != "bicubic" else 0,
    }


def _hdr_profile_params(profile: str) -> tuple[float, float, float, float]:
    """Return (shadow_lift, local_contrast, highlight_rolloff, vibrance)."""
    return {
        "natural":   (0.025, 0.16, 0.72, 0.10),
        "balanced":  (0.035, 0.24, 0.82, 0.16),
        "cinematic": (0.020, 0.32, 0.92, 0.12),
        "punchy":    (0.045, 0.42, 0.76, 0.24),
    }.get(str(profile), (0.035, 0.24, 0.82, 0.16))


def _aetherscale_hdr_frame(
    frame_chw: torch.Tensor,
    *,
    profile: str,
    strength: float,
    saturation: float,
    contrast: float,
    highlight_preservation: float,
) -> torch.Tensor:
    """High-quality SDR enhancement in linear-ish working space.

    This intentionally stays inside ComfyUI's normalized IMAGE range. It is an
    HDR-style enhancer/tone mapper, not an HDR10/PQ metadata encoder.
    """
    x = frame_chw.float().clamp(0.0, 1.0)
    s = max(0.0, float(strength))
    sat = max(0.0, float(saturation))
    ctr = max(0.0, float(contrast))
    hp = max(0.0, float(highlight_preservation))
    shadow_lift, local_contrast, rolloff, vibrance = _hdr_profile_params(profile)

    # Approximate display->linear transform for luminance-aware operations.
    lin = torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).pow(2.4),
    )
    luma = (lin[0:1] * 0.2126 + lin[1:2] * 0.7152 + lin[2:3] * 0.0722).clamp_min(1e-6)

    # Wide, smooth local adaptation. Small fixed kernel keeps memory bounded.
    base = F.avg_pool2d(luma.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
    detail = luma - base

    # Lift deep shadows, add controlled local contrast, then compress highlights.
    lifted = luma + s * shadow_lift * (1.0 - luma).pow(2.0)
    adapted = lifted + detail * (s * local_contrast * ctr)
    adapted = adapted.clamp_min(0.0)

    # Shoulder strength increases with preservation: bright detail is compressed
    # smoothly instead of clipping.
    shoulder = 1.0 + s * rolloff * (0.35 + 0.65 * hp)
    mapped_luma = adapted / (adapted + shoulder * (1.0 - adapted).clamp_min(0.0) + 1e-6)
    mapped_luma = mapped_luma.clamp(0.0, 1.0)

    # Preserve chroma ratios while remapping luminance.
    ratio = (mapped_luma / luma).clamp(0.0, 8.0)
    lin = (lin * ratio).clamp(0.0, 1.0)

    # Contrast around perceptual middle grey in linear space.
    mid = 0.18
    lin = ((lin - mid) * (1.0 + (ctr - 1.0) * 0.70 * s) + mid).clamp(0.0, 1.0)

    # Luminance-preserving saturation + mild vibrance for low-saturation colors.
    lum2 = (lin[0:1] * 0.2126 + lin[1:2] * 0.7152 + lin[2:3] * 0.0722)
    chroma = lin - lum2
    chroma_mag = chroma.abs().mean(dim=0, keepdim=True)
    vib_gain = 1.0 + s * vibrance * (1.0 - (chroma_mag * 4.0).clamp(0.0, 1.0))
    lin = (lum2 + chroma * sat * vib_gain).clamp(0.0, 1.0)

    # Linear->display transfer.
    out = torch.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * lin.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055,
    )
    return out.clamp(0.0, 1.0)


def _stream_cuda_hdr(
    images_bhwc: torch.Tensor,
    *,
    config: VFXConfig,
    memory_policy: str,
    output_device: str,
    output_precision: str = "auto",
    output_storage: str = "auto",
    clean_cache: bool = True,
) -> tuple[torch.Tensor, dict]:
    src_device = images_bhwc.device
    cuda_device = torch.device(f"cuda:{config.device}")
    batch = int(images_bhwc.shape[0])
    in_channels = int(images_bhwc.shape[-1])
    out_channels = 4 if in_channels > 3 else 3
    extras = dict(config.extras)
    profile = extras.get("mode_profile", "balanced")
    strength = float(extras.get("strength", 0.75))
    saturation = float(extras.get("saturation", 1.0))
    contrast = float(extras.get("contrast", 1.0))
    highlight_preservation = float(extras.get("highlight_preservation", 0.75))

    out_shape = (batch, config.out_height, config.out_width, out_channels)
    out_dtype = resolve_dtype(
        requested=str(output_precision),
        shape=out_shape,
        input_dtype=images_bhwc.dtype,
        auto_fp16_threshold_mb=512,
    )
    out_cpu, storage = allocate_cpu_tensor(
        out_shape,
        dtype=out_dtype,
        storage_mode=str(output_storage),
        prefix="hdr",
        mmap_threshold_mb=512,
        clean_cache=bool(clean_cache),
    )
    console_progress = ConsoleProgress("HDR", batch, unit="frame")

    with torch.cuda.device(cuda_device):
        for i in range(batch):
            frame = images_bhwc[i, ..., :3].to(
                device=cuda_device, dtype=torch.float32, non_blocking=False
            ).permute(2, 0, 1).contiguous()
            enhanced = _aetherscale_hdr_frame(
                frame,
                profile=profile,
                strength=strength,
                saturation=saturation,
                contrast=contrast,
                highlight_preservation=highlight_preservation,
            )
            rgb = enhanced.permute(1, 2, 0).to(dtype=out_dtype)
            out_cpu[i, ..., :3].copy_(rgb.to("cpu", non_blocking=False), non_blocking=False)
            if out_channels == 4:
                alpha = images_bhwc[i, ..., 3:4].detach()
                if alpha.device.type != "cpu":
                    alpha = alpha.to("cpu", non_blocking=False)
                out_cpu[i, ..., 3:4].copy_(alpha.to(dtype=out_dtype), non_blocking=False)
            del rgb, enhanced, frame
            if memory_policy == "aggressive":
                torch.cuda.empty_cache()
            sync_file_backed_tensor(
                out_cpu,
                bytes_written=out_cpu[i].numel() * out_cpu.element_size(),
            )
            console_progress.update(1)

    sync_file_backed_tensor(out_cpu, force=True)
    console_progress.close(status="done")
    storage_stats = {
        "output_dtype": str(out_dtype).replace("torch.", ""),
        "output_storage_backend": storage.backend,
        "output_storage_bytes": int(storage.bytes),
        "output_storage_gib": round(storage.bytes / (1024 ** 3), 3),
        "output_storage_path": storage.path,
        "output_storage_fallback_reason": storage.fallback_reason,
        "clean_cache": bool(clean_cache),
    }
    return _finalize_output_device(out_cpu, src_device, output_device), storage_stats

def _stream_video_hdr(
    images_bhwc: torch.Tensor,
    *,
    config: VFXConfig,
    cache_policy: str,
    cuda_stream_mode: str,
    memory_policy: str,
    output_device: str,
    output_precision: str = "auto",
    output_storage: str = "auto",
    clean_cache: bool = True,
) -> tuple[torch.Tensor, dict]:
    src_device = images_bhwc.device
    cuda_device = torch.device(f"cuda:{config.device}")
    batch = int(images_bhwc.shape[0])
    in_channels = int(images_bhwc.shape[-1])
    out_channels = 4 if in_channels > 3 else 3

    if cache_policy == "single":
        _CACHE.evict_except(config)
    elif cache_policy == "none":
        _CACHE.clear()

    effect = _CACHE.get(config)
    out_shape = (batch, config.out_height, config.out_width, out_channels)
    out_dtype = resolve_dtype(
        requested=str(output_precision),
        shape=out_shape,
        input_dtype=images_bhwc.dtype,
        auto_fp16_threshold_mb=512,
    )
    out_cpu, storage = allocate_cpu_tensor(
        out_shape,
        dtype=out_dtype,
        storage_mode=str(output_storage),
        prefix="hdr_vfx",
        mmap_threshold_mb=512,
        clean_cache=bool(clean_cache),
    )
    console_progress = ConsoleProgress("HDR / NVIDIA VFX", batch, unit="frame")

    with torch.cuda.device(cuda_device):
        if cuda_stream_mode == "dedicated":
            stream = torch.cuda.Stream(device=cuda_device)
        else:
            stream = torch.cuda.current_stream(device=cuda_device)

        for i in range(batch):
            frame_hwc = images_bhwc[i, ..., :3]
            frame = (
                frame_hwc.to(device=cuda_device, dtype=torch.float32, non_blocking=False)
                .clamp_(0.0, 1.0).permute(2, 0, 1).contiguous()
            )
            result = effect.run(frame, non_blocking=False, stream_ptr=int(stream.cuda_stream))
            out_view_chw = torch.from_dlpack(result.image)
            out_view_hwc = out_view_chw.permute(1, 2, 0).to(dtype=out_dtype)
            out_cpu[i, ..., :3].copy_(out_view_hwc.to("cpu", non_blocking=False), non_blocking=False)
            if out_channels == 4:
                alpha = images_bhwc[i, ..., 3:4].detach()
                if alpha.device.type != "cpu":
                    alpha = alpha.to("cpu", non_blocking=False)
                out_cpu[i, ..., 3:4].copy_(alpha.to(dtype=out_dtype), non_blocking=False)
            del out_view_hwc, out_view_chw, result, frame, frame_hwc
            if memory_policy == "aggressive":
                torch.cuda.empty_cache()
            sync_file_backed_tensor(
                out_cpu,
                bytes_written=out_cpu[i].numel() * out_cpu.element_size(),
            )
            console_progress.update(1)

    sync_file_backed_tensor(out_cpu, force=True)
    console_progress.close(status="done")
    if cache_policy == "none":
        _CACHE.clear()

    storage_stats = {
        "output_dtype": str(out_dtype).replace("torch.", ""),
        "output_storage_backend": storage.backend,
        "output_storage_bytes": int(storage.bytes),
        "output_storage_gib": round(storage.bytes / (1024 ** 3), 3),
        "output_storage_path": storage.path,
        "output_storage_fallback_reason": storage.fallback_reason,
        "clean_cache": bool(clean_cache),
    }
    return _finalize_output_device(out_cpu, src_device, output_device), storage_stats

def available_capabilities() -> Dict[str, Any]:
    state = RuntimeManager.probe()
    payload: Dict[str, Any] = {
        "runtime_ready": state.ready,
        "installed_version": state.installed_version,
        "video_super_res": False,
        "video_hdr": False,
        "symbols": [],
    }
    if not state.ready:
        return payload

    try:
        import nvvfx
    except Exception as exc:
        payload["runtime_import_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    symbols = sorted(dir(nvvfx))
    payload["symbols"] = symbols
    payload["video_super_res"] = hasattr(nvvfx, "VideoSuperRes")
    payload["video_hdr"] = _find_first_attr(nvvfx, HDR_CLASS_CANDIDATES) is not None
    payload["hdr_symbol_candidates"] = [name for name in symbols if "hdr" in name.lower()]
    return payload


class VFXBackend:
    @staticmethod
    def clear_effect_cache() -> None:
        _CACHE.clear()

    @staticmethod
    def effect_cache_size() -> int:
        return _CACHE.size()

    @staticmethod
    def available_capabilities() -> Dict[str, Any]:
        return available_capabilities()

    @staticmethod
    def run_video_super_res(
        images_bhwc: torch.Tensor,
        *,
        config: VFXConfig,
        cache_policy: str,
        cuda_stream_mode: str,
        memory_policy: str,
        vram_guard: str,
        min_free_vram_mb: int,
        output_device: str,
        output_precision: str = "auto",
        output_storage: str = "auto",
        clean_cache: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        if images_bhwc.ndim != 4:
            raise ValueError(
                f"Expected ComfyUI IMAGE tensor [B,H,W,C], got {tuple(images_bhwc.shape)}"
            )
        if images_bhwc.shape[-1] < 3:
            raise ValueError("AetherScale requires at least 3 image channels (RGB).")
        if not torch.cuda.is_available():
            raise RuntimeError("AetherScale requires an NVIDIA CUDA GPU.")
        device_count = torch.cuda.device_count()
        if config.device < 0 or config.device >= device_count:
            raise ValueError(
                f"CUDA device {config.device} is invalid; detected {device_count} CUDA device(s)."
            )

        t0 = time.perf_counter()
        batch = int(images_bhwc.shape[0])
        in_h = int(images_bhwc.shape[1])
        in_w = int(images_bhwc.shape[2])

        if config.mode in ("denoise", "deblur"):
            if (config.out_width, config.out_height) != (in_w, in_h):
                raise ValueError(
                    f"{config.mode} is same-resolution only. "
                    f"Requested {config.out_width}x{config.out_height} for {in_w}x{in_h} input."
                )
        elif config.mode != "bicubic":
            ratio_w = config.out_width / in_w
            ratio_h = config.out_height / in_h
            if ratio_w > 4.0 + 1e-6 or ratio_h > 4.0 + 1e-6:
                raise ValueError(
                    f"NVIDIA VSR supports up to 4x per dimension; requested "
                    f"{ratio_w:.3f}x × {ratio_h:.3f}x."
                )

        before_free, after_release_free, total_vram, release_reason, release_actions = _prepare_vram(
            vram_guard=vram_guard,
            min_free_vram_mb=min_free_vram_mb,
            device=config.device,
        )

        out, storage_stats = _stream_video_super_res(
            images_bhwc,
            config=config,
            cache_policy=cache_policy,
            cuda_stream_mode=cuda_stream_mode,
            memory_policy=memory_policy,
            output_device=output_device,
            output_precision=output_precision,
            output_storage=output_storage,
            clean_cache=bool(clean_cache),
        )

        if memory_policy in ("balanced", "aggressive"):
            gc.collect()
        if memory_policy == "aggressive":
            torch.cuda.empty_cache()

        final_free, _ = _cuda_memory(config.device)
        elapsed = time.perf_counter() - t0
        stats = {
            "engine": "NVIDIA VFX VideoSuperRes" if config.mode != "bicubic" else "PyTorch Bicubic",
            "effect_type": config.effect_type,
            "mode": config.mode,
            "quality": config.quality,
            "output": [config.out_width, config.out_height],
            "frames": batch,
            "seconds": elapsed,
            "ms_per_frame": elapsed * 1000.0 / max(1, batch),
            "effect_cache_entries": _CACHE.size(),
            "cuda_device": config.device,
            "streaming_input": True,
            "zero_copy_dlpack_view": config.mode != "bicubic",
            "cuda_output_clone": False,
            "output_device": str(out.device),
            "clean_cache": bool(clean_cache),
            "vram_guard": vram_guard,
            "vram_release_reason": release_reason,
            "vram_release_actions": release_actions,
            "vram_free_before_mb": round(before_free / (1024 * 1024), 1),
            "vram_free_after_release_mb": round(after_release_free / (1024 * 1024), 1),
            "vram_free_final_mb": round(final_free / (1024 * 1024), 1),
            "vram_total_mb": round(total_vram / (1024 * 1024), 1),
            **storage_stats,
        }
        return out, stats

    @staticmethod
    def run_video_hdr(
        images_bhwc: torch.Tensor,
        *,
        config: VFXConfig,
        cache_policy: str,
        cuda_stream_mode: str,
        memory_policy: str,
        vram_guard: str,
        min_free_vram_mb: int,
        output_device: str,
        output_precision: str = "auto",
        output_storage: str = "auto",
        clean_cache: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        if images_bhwc.ndim != 4:
            raise ValueError(
                f"Expected ComfyUI IMAGE tensor [B,H,W,C], got {tuple(images_bhwc.shape)}"
            )
        if images_bhwc.shape[-1] < 3:
            raise ValueError("AetherScale requires at least 3 image channels (RGB).")
        if not torch.cuda.is_available():
            raise RuntimeError("AetherScale requires an NVIDIA CUDA GPU.")
        device_count = torch.cuda.device_count()
        if config.device < 0 or config.device >= device_count:
            raise ValueError(
                f"CUDA device {config.device} is invalid; detected {device_count} CUDA device(s)."
            )

        t0 = time.perf_counter()
        batch = int(images_bhwc.shape[0])

        before_free, after_release_free, total_vram, release_reason, release_actions = _prepare_vram(
            vram_guard=vram_guard,
            min_free_vram_mb=min_free_vram_mb,
            device=config.device,
        )

        nvvfx = _ensure_runtime_module()
        native_hdr = _find_first_attr(nvvfx, HDR_CLASS_CANDIDATES) is not None
        if native_hdr:
            out, storage_stats = _stream_video_hdr(
                images_bhwc,
                config=config,
                cache_policy=cache_policy,
                cuda_stream_mode=cuda_stream_mode,
                memory_policy=memory_policy,
                output_device=output_device,
                output_precision=output_precision,
                output_storage=output_storage,
                clean_cache=clean_cache,
            )
            hdr_engine = "NVIDIA VFX HDR"
        else:
            # Current NVIDIA VFX SDK releases do not expose an HDR effect.
            # Use AetherScale's CUDA-native HDR-style enhancer instead of
            # failing symbol discovery.
            out, storage_stats = _stream_cuda_hdr(
                images_bhwc,
                config=config,
                memory_policy=memory_policy,
                output_device=output_device,
                output_precision=output_precision,
                output_storage=output_storage,
                clean_cache=clean_cache,
            )
            hdr_engine = "AetherScale CUDA HDR"

        if memory_policy in ("balanced", "aggressive"):
            gc.collect()
        if memory_policy == "aggressive":
            torch.cuda.empty_cache()

        final_free, _ = _cuda_memory(config.device)
        elapsed = time.perf_counter() - t0
        stats = {
            "engine": hdr_engine,
            "native_nvidia_vfx_hdr_available": native_hdr,
            "hdr_output_note": "Normalized ComfyUI IMAGE enhancement; not HDR10/PQ metadata encoding",
            "effect_type": config.effect_type,
            "mode": config.mode,
            "quality": config.quality,
            "extras": dict(config.extras),
            "output": [config.out_width, config.out_height],
            "frames": batch,
            "seconds": elapsed,
            "ms_per_frame": elapsed * 1000.0 / max(1, batch),
            "effect_cache_entries": _CACHE.size(),
            "cuda_device": config.device,
            "streaming_input": True,
            "zero_copy_dlpack_view": True,
            "cuda_output_clone": False,
            "output_device": str(out.device),
            "vram_guard": vram_guard,
            "vram_release_reason": release_reason,
            "vram_release_actions": release_actions,
            "vram_free_before_mb": round(before_free / (1024 * 1024), 1),
            "vram_free_after_release_mb": round(after_release_free / (1024 * 1024), 1),
            "vram_free_final_mb": round(final_free / (1024 * 1024), 1),
            "vram_total_mb": round(total_vram / (1024 * 1024), 1),
            **storage_stats,
        }
        return out, stats
