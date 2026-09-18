from __future__ import annotations

import json
import platform
import subprocess
from typing import Any, Dict

import torch

from .backend.runtime import RuntimeManager
from .backend.vfx import VFXBackend, VFXConfig, resolve_output_size
from .backend.neural import MotionPacket, analyze_motion, flow_visualization, neural_vram_plan
from .backend.carrier import process_carrier, ensure_carrier, CarrierError, carrier_gpu_choices
from .backend.mfg import generate_mfg_lab, encode_mfg_video
from .backend.combine import encode_video_batch, video_gpu_choices
from .backend.video_loader import list_input_videos, load_video_low_ram, resolve_video_path
from .backend.progress import console_node
from .backend.nr_interop import (
    NativeNRInteropError,
    ensure_native_bundle,
    process_native_interop,
    runtime_info as native_nr_runtime_info,
    shutdown as shutdown_native_nr,
)
from .backend.dlssnr import (
    DLSSNRError, ensure_bridge as ensure_dlss5_bridge, probe as probe_dlss5,
    process as process_dlss5, runtime_info as dlss5_runtime_info, shutdown as shutdown_dlss5,
)


def _driver_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return torch.cuda.get_device_name(0)


def _common_runtime_inputs() -> Dict[str, tuple]:
    device_max = max(0, torch.cuda.device_count() - 1) if torch.cuda.is_available() else 0
    return {
        "cuda_device": (
            "INT",
            {"default": 0, "min": 0, "max": device_max, "step": 1},
        ),
        "effect_cache": (
            ["single", "persistent", "none"],
            {"default": "single"},
        ),
        "cuda_stream": (
            ["current", "dedicated"],
            {"default": "current"},
        ),
        "memory_policy": (
            ["performance", "balanced", "aggressive"],
            {"default": "performance"},
        ),
        "vram_guard": (
            ["auto", "release_models", "preserve_models"],
            {"default": "auto"},
        ),
        "min_free_vram_mb": (
            "INT",
            {"default": 2048, "min": 0, "max": 24576, "step": 128},
        ),
        "output_device": (
            ["cpu_safe", "same_as_input"],
            {"default": "cpu_safe"},
        ),
        "auto_bootstrap": ("BOOLEAN", {"default": True}),
    }


def _ensure_runtime_if_needed(auto_bootstrap: bool, node_name: str) -> None:
    if auto_bootstrap:
        return
    state = RuntimeManager.probe()
    if not state.ready:
        raise RuntimeError(
            f"{node_name}: NVIDIA VFX runtime is missing and auto_bootstrap is disabled. "
            "Enable auto_bootstrap or use the AetherScale Runtime node."
        )


class AetherScaleSuperResolution:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",),
            "source_profile": (
                ["high_bitrate", "compressed", "bicubic"],
                {"default": "high_bitrate"},
            ),
            "quality": (
                ["ultra", "high", "medium", "low"],
                {"default": "high"},
            ),
            "resize_mode": (
                ["scale", "exact", "long_edge"],
                {"default": "scale"},
            ),
            "scale": (
                "FLOAT",
                {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05},
            ),
            "target_width": (
                "INT",
                {"default": 3840, "min": 0, "max": 16384, "step": 8},
            ),
            "target_height": (
                "INT",
                {"default": 2160, "min": 0, "max": 16384, "step": 8},
            ),
            "long_edge": (
                "INT",
                {"default": 3840, "min": 64, "max": 16384, "step": 8},
            ),
            "dimension_alignment": (
                ["1", "2", "4", "8", "16", "32", "64"],
                {"default": "8"},
            ),
        }
        req.update(_common_runtime_inputs())
        optional = {
            "output_precision": (["auto", "float16", "float32"], {"default": "auto"}),
            "output_storage": (["auto", "mmap", "ram"], {"default": "auto"}),
            "clean_cache": ("BOOLEAN", {"default": True}),
        }
        return {"required": req, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/Enhance"
    DESCRIPTION = "Upscaling node for clean/high-bitrate or compressed sources with VRAM guard."

    def run(
        self,
        image: torch.Tensor,
        source_profile: str,
        quality: str,
        resize_mode: str,
        scale: float,
        target_width: int,
        target_height: int,
        long_edge: int,
        dimension_alignment: str,
        cuda_device: int,
        effect_cache: str,
        cuda_stream: str,
        memory_policy: str,
        vram_guard: str,
        min_free_vram_mb: int,
        output_device: str,
        auto_bootstrap: bool,
        output_precision: str = "auto",
        output_storage: str = "auto",
        clean_cache: bool = True,
    ):
        _ensure_runtime_if_needed(auto_bootstrap, "AetherScale Super Resolution")
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
        h, w = int(image.shape[1]), int(image.shape[2])

        out_w, out_h = resolve_output_size(
            width=w,
            height=h,
            resize_mode=resize_mode,
            scale=float(scale),
            target_width=int(target_width),
            target_height=int(target_height),
            long_edge=int(long_edge),
            alignment=int(dimension_alignment),
        )

        config = VFXConfig(
            effect_type="video_super_res",
            mode=source_profile,
            quality=quality,
            out_width=out_w,
            out_height=out_h,
            device=int(cuda_device),
        )
        result, stats = VFXBackend.run_video_super_res(
            image,
            config=config,
            cache_policy=effect_cache,
            cuda_stream_mode=cuda_stream,
            memory_policy=memory_policy,
            vram_guard=vram_guard,
            min_free_vram_mb=int(min_free_vram_mb),
            output_device=output_device,
            output_precision=output_precision,
            output_storage=output_storage,
            clean_cache=bool(clean_cache),
        )
        return (result, json.dumps(stats, indent=2))


class AetherScaleRestoration:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",),
            "task": (
                ["artifact_reduction", "ai_denoise", "ai_deblur"],
                {"default": "artifact_reduction"},
            ),
            "quality": (
                ["ultra", "high", "medium", "low"],
                {"default": "high"},
            ),
        }
        req.update(_common_runtime_inputs())
        return {"required": req}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/Restore"
    DESCRIPTION = "Same-resolution restoration grouped into one node: artifact reduction, denoise, deblur."

    def run(
        self,
        image: torch.Tensor,
        task: str,
        quality: str,
        cuda_device: int,
        effect_cache: str,
        cuda_stream: str,
        memory_policy: str,
        vram_guard: str,
        min_free_vram_mb: int,
        output_device: str,
        auto_bootstrap: bool,
    ):
        _ensure_runtime_if_needed(auto_bootstrap, "AetherScale Restoration")
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
        h, w = int(image.shape[1]), int(image.shape[2])

        mode = {
            "artifact_reduction": "compressed",
            "ai_denoise": "denoise",
            "ai_deblur": "deblur",
        }[task]

        config = VFXConfig(
            effect_type="video_super_res",
            mode=mode,
            quality=quality,
            out_width=w,
            out_height=h,
            device=int(cuda_device),
        )
        result, stats = VFXBackend.run_video_super_res(
            image,
            config=config,
            cache_policy=effect_cache,
            cuda_stream_mode=cuda_stream,
            memory_policy=memory_policy,
            vram_guard=vram_guard,
            min_free_vram_mb=int(min_free_vram_mb),
            output_device=output_device,
        )
        stats["requested_task"] = task
        return (result, json.dumps(stats, indent=2))


class AetherScaleHDR:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",),
            "quality": (
                ["ultra", "high", "medium", "low"],
                {"default": "high"},
            ),
            "mode_profile": (
                ["balanced", "cinematic", "punchy", "natural"],
                {"default": "balanced"},
            ),
            "strength": (
                "FLOAT",
                {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05},
            ),
            "saturation": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
            ),
            "contrast": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
            ),
            "highlight_preservation": (
                "FLOAT",
                {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05},
            ),
        }
        req.update(_common_runtime_inputs())
        return {
            "required": req,
            "optional": {
                "output_precision": (["auto", "float16", "float32"], {"default": "auto"}),
                "output_storage": (["auto", "mmap", "ram"], {"default": "auto"}),
                "clean_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/HDR"
    DESCRIPTION = "CUDA HDR-style enhancer with automatic NVIDIA VFX HDR use if a future runtime exposes it."

    def run(
        self,
        image: torch.Tensor,
        quality: str,
        mode_profile: str,
        strength: float,
        saturation: float,
        contrast: float,
        highlight_preservation: float,
        cuda_device: int,
        effect_cache: str,
        cuda_stream: str,
        memory_policy: str,
        vram_guard: str,
        min_free_vram_mb: int,
        output_device: str,
        auto_bootstrap: bool,
        output_precision: str = "auto",
        output_storage: str = "auto",
        clean_cache: bool = True,
    ):
        _ensure_runtime_if_needed(auto_bootstrap, "AetherScale HDR")
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
        h, w = int(image.shape[1]), int(image.shape[2])

        extras = tuple(
            sorted(
                {
                    "mode_profile": str(mode_profile),
                    "strength": f"{float(strength):.6f}",
                    "saturation": f"{float(saturation):.6f}",
                    "contrast": f"{float(contrast):.6f}",
                    "highlight_preservation": f"{float(highlight_preservation):.6f}",
                }.items()
            )
        )
        config = VFXConfig(
            effect_type="video_hdr",
            mode="hdr",
            quality=quality,
            out_width=w,
            out_height=h,
            device=int(cuda_device),
            extras=extras,
        )
        result, stats = VFXBackend.run_video_hdr(
            image,
            config=config,
            cache_policy=effect_cache,
            cuda_stream_mode=cuda_stream,
            memory_policy=memory_policy,
            vram_guard=vram_guard,
            min_free_vram_mb=int(min_free_vram_mb),
            output_device=output_device,
            output_precision=str(output_precision),
            output_storage=str(output_storage),
            clean_cache=bool(clean_cache),
        )
        return (result, json.dumps(stats, indent=2))


class AetherScaleRuntime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (
                    ["status", "install_or_update", "repair", "clear_effect_cache", "install_native_nr", "shutdown_native_nr", "install_dlss5_bridge", "shutdown_dlss5", "clear_runtime"],
                    {"default": "status"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "AetherScale/System"
    DESCRIPTION = "Inspect, bootstrap, repair, or clear AetherScale's private NVIDIA runtime."

    def run(self, action: str):
        if action == "install_or_update":
            state = RuntimeManager.ensure(force_reinstall=False)
        elif action == "repair":
            VFXBackend.clear_effect_cache()
            state = RuntimeManager.ensure(force_reinstall=True)
        elif action == "clear_effect_cache":
            VFXBackend.clear_effect_cache()
            state = RuntimeManager.probe()
        elif action == "install_native_nr":
            ensure_native_bundle(auto_bootstrap=True, gpu_index=0)
            state = RuntimeManager.probe()
        elif action == "shutdown_native_nr":
            shutdown_native_nr()
            state = RuntimeManager.probe()
        elif action == "install_dlss5_bridge":
            ensure_dlss5_bridge(force=False)
            state = RuntimeManager.probe()
        elif action == "shutdown_dlss5":
            shutdown_dlss5()
            state = RuntimeManager.probe()
        elif action == "clear_runtime":
            VFXBackend.clear_effect_cache()
            state = RuntimeManager.clear()
        else:
            state = RuntimeManager.probe()

        info = {
            "runtime": state.__dict__ if hasattr(state, "__dict__") else {
                "ready": state.ready,
                "package": state.package,
                "requested_version": state.requested_version,
                "installed_version": state.installed_version,
                "vendor_path": state.vendor_path,
                "python": state.python,
                "platform": state.platform,
                "message": state.message,
            },
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu": _driver_info(),
            "effect_cache_entries": VFXBackend.effect_cache_size(),
            "capabilities": VFXBackend.available_capabilities(),
            "neural_rendering": {
                "native_interop": native_nr_runtime_info(),
                "legacy_direct": dlss5_runtime_info(),
            },
        }
        return (json.dumps(info, indent=2),)




class AetherScaleMFGLab:
    @classmethod
    def INPUT_TYPES(cls):
        device_max = max(0, torch.cuda.device_count() - 1) if torch.cuda.is_available() else 0
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (
                    ["native_dlssg", "surrogate_mv", "probe_only"],
                    {"default": "native_dlssg"},
                ),
                "multiplier": (
                    ["2x", "3x", "4x", "5x", "6x"],
                    {"default": "4x"},
                ),
                "motion_source": (
                    ["internal_dis", "connected_motion", "internal_compact", "rgb_only"],
                    {"default": "internal_dis"},
                ),
                "surrogate_method": (
                    ["mv_blend", "linear_blend", "frame_repeat"],
                    {"default": "mv_blend"},
                ),
                "scene_cut_strategy": (
                    ["repeat_previous", "repeat_current", "linear_blend"],
                    {"default": "repeat_previous"},
                ),
                "cuda_device": (
                    "INT", {"default": 0, "min": 0, "max": device_max, "step": 1},
                ),
                "output_device": (
                    ["cpu_safe", "same_as_input"], {"default": "cpu_safe"},
                ),
            },
            "optional": {
                "motion": ("AETHERSCALE_MOTION",),
                "output_precision": (
                    ["auto", "float16", "float32"], {"default": "auto"},
                ),
                "output_storage": (
                    ["auto", "mmap", "ram"], {"default": "auto"},
                ),
                "clean_cache": ("BOOLEAN", {"default": True}),
                "artifact_guard": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "emissive_protection": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thin_detail_protection": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mv_confidence_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fallback_mode": (["linear_blend", "closest_source"], {"default": "linear_blend"}),
                "synthesis_mode": (["continuous_temporal", "legacy_guarded"], {"default": "continuous_temporal"}),
                "source_frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
                "native_auto_bootstrap": ("BOOLEAN", {"default": True}),
                "native_guide_source": (["internal_dis", "connected_motion", "zero_motion"], {"default": "internal_dis"}),
                "native_scene_cut_threshold": ("FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/MFG"
    DESCRIPTION = (
        "Native NVIDIA DLSS Frame Generation for ComfyUI IMAGE sequences. "
        "Uses a D3D12 DLSSG worker and DIS motion guides; the previous surrogate remains available as a fallback/debug backend."
    )

    def run(
        self,
        images,
        mode,
        multiplier,
        motion_source,
        surrogate_method,
        scene_cut_strategy,
        cuda_device,
        output_device,
        motion=None,
        output_precision="auto",
        output_storage="auto",
        clean_cache=True,
        artifact_guard=1.0,
        emissive_protection=0.85,
        thin_detail_protection=0.75,
        mv_confidence_threshold=0.45,
        fallback_mode="linear_blend",
        synthesis_mode="continuous_temporal",
        source_frame_rate=24.0,
        native_auto_bootstrap=True,
        native_guide_source="internal_dis",
        native_scene_cut_threshold=0.24,
    ):
        multiplier_int = {"2x": 2, "3x": 3, "4x": 4, "5x": 5, "6x": 6}[str(multiplier)]
        resolved_motion = None if motion_source == "rgb_only" else motion
        result, stats = generate_mfg_lab(
            images,
            motion=resolved_motion,
            mode=str(mode),
            motion_source=str(motion_source),
            multiplier=int(multiplier_int),
            surrogate_method=str(surrogate_method),
            scene_cut_strategy=str(scene_cut_strategy),
            cuda_device=int(cuda_device),
            output_device=str(output_device),
            output_precision=str(output_precision),
            output_storage=str(output_storage),
            clean_cache=bool(clean_cache),
            artifact_guard=float(artifact_guard),
            emissive_protection=float(emissive_protection),
            thin_detail_protection=float(thin_detail_protection),
            mv_confidence_threshold=float(mv_confidence_threshold),
            fallback_mode=str(fallback_mode),
            synthesis_mode=str(synthesis_mode),
            source_frame_rate=float(source_frame_rate),
            native_auto_bootstrap=bool(native_auto_bootstrap),
            native_guide_source=str(native_guide_source),
            native_scene_cut_threshold=float(native_scene_cut_threshold),
        )
        return (result, json.dumps(stats, indent=2))



class AetherScaleMFGVideo:
    @classmethod
    def INPUT_TYPES(cls):
        device_max = max(0, torch.cuda.device_count() - 1) if torch.cuda.is_available() else 0
        gpu_choices, gpu_default = video_gpu_choices()
        return {
            "required": {
                "images": ("IMAGE",),
                "source_frame_rate": (
                    "FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01},
                ),
                "multiplier": (
                    ["2x", "3x", "4x", "5x", "6x"], {"default": "4x"},
                ),
                "motion_source": (
                    ["internal_dis", "connected_motion", "internal_compact", "rgb_only"],
                    {"default": "internal_dis"},
                ),
                "surrogate_method": (
                    ["mv_blend", "linear_blend", "frame_repeat"], {"default": "mv_blend"},
                ),
                "scene_cut_strategy": (
                    ["repeat_previous", "repeat_current", "linear_blend"],
                    {"default": "repeat_previous"},
                ),
                "codec": (
                    ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "prores_proxy", "prores_lt", "prores_standard", "prores_hq", "prores_4444", "prores_4444_xq"],
                    {"default": "h264_nvenc"},
                ),
                "preset": (
                    ["p1", "p2", "p3", "p4", "p5", "p6", "p7"], {"default": "p4"},
                ),
                "bitrate_mbps": (
                    "INT", {"default": 16, "min": 1, "max": 500, "step": 1},
                ),
                "pixel_format": (
                    ["yuv420p", "yuv444p"], {"default": "yuv420p"},
                ),
                "filename_prefix": (
                    "STRING", {"default": "AetherScale/MFG", "multiline": False},
                ),
                "cuda_device": (
                    "INT", {"default": 0, "min": 0, "max": device_max, "step": 1},
                ),
            },
            "optional": {
                "motion": ("AETHERSCALE_MOTION",),
                "artifact_guard": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "emissive_protection": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thin_detail_protection": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mv_confidence_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fallback_mode": (["linear_blend", "closest_source"], {"default": "linear_blend"}),
                "synthesis_mode": (["continuous_temporal", "legacy_guarded"], {"default": "continuous_temporal"}),
                "backend": (["native_dlssg", "surrogate_mv"], {"default": "native_dlssg"}),
                "native_auto_bootstrap": ("BOOLEAN", {"default": True}),
                "native_guide_source": (["internal_dis", "connected_motion", "zero_motion"], {"default": "internal_dis"}),
                "native_scene_cut_threshold": ("FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01}),
                "nvenc_gpu": (gpu_choices, {"default": gpu_default}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/MFG"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Native NVIDIA DLSS Frame Generation streamed directly into FFmpeg/NVENC. "
        "No full multiplied IMAGE batch is materialized; the legacy surrogate remains selectable for comparison."
    )

    def run(
        self,
        images,
        source_frame_rate,
        multiplier,
        motion_source,
        surrogate_method,
        scene_cut_strategy,
        codec,
        preset,
        bitrate_mbps,
        pixel_format,
        filename_prefix,
        cuda_device,
        motion=None,
        artifact_guard=1.0,
        emissive_protection=0.85,
        thin_detail_protection=0.75,
        mv_confidence_threshold=0.45,
        fallback_mode="linear_blend",
        synthesis_mode="continuous_temporal",
        backend="native_dlssg",
        native_auto_bootstrap=True,
        native_guide_source="internal_dis",
        native_scene_cut_threshold=0.24,
        nvenc_gpu="auto",
    ):
        multiplier_int = {"2x": 2, "3x": 3, "4x": 4, "5x": 5, "6x": 6}[str(multiplier)]
        resolved_motion = None if motion_source == "rgb_only" else motion
        stats = encode_mfg_video(
            images,
            motion=resolved_motion,
            source_frame_rate=float(source_frame_rate),
            multiplier=int(multiplier_int),
            motion_source=str(motion_source),
            surrogate_method=str(surrogate_method),
            scene_cut_strategy=str(scene_cut_strategy),
            cuda_device=int(cuda_device),
            codec=str(codec),
            preset=str(preset),
            bitrate_mbps=int(bitrate_mbps),
            filename_prefix=str(filename_prefix),
            pixel_format=str(pixel_format),
            artifact_guard=float(artifact_guard),
            emissive_protection=float(emissive_protection),
            thin_detail_protection=float(thin_detail_protection),
            mv_confidence_threshold=float(mv_confidence_threshold),
            fallback_mode=str(fallback_mode),
            synthesis_mode=str(synthesis_mode),
            backend=str(backend),
            native_auto_bootstrap=bool(native_auto_bootstrap),
            native_guide_source=str(native_guide_source),
            native_scene_cut_threshold=float(native_scene_cut_threshold),
            nvenc_gpu=str(nvenc_gpu),
        )
        return (str(stats["output_path"]), json.dumps(stats, indent=2))




class AetherScaleVideoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (list_input_videos(),),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "precision": (["auto", "float16", "float32"], {"default": "auto"}),
                "decode_chunk_frames": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "load_audio": ("BOOLEAN", {"default": True}),
                "clean_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "path_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "audio", "video_info", "frame_rate", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/IO"
    DESCRIPTION = (
        "Low-RAM FFmpeg video loader. Decodes small chunks and writes them sequentially "
        "to an FP16/FP32 file-backed IMAGE tensor instead of materializing the whole video "
        "as a giant float32 RAM batch."
    )

    @classmethod
    def IS_CHANGED(cls, video, path_override="", **kwargs):
        try:
            path = resolve_video_path(video, path_override)
            st = path.stat()
            return f"{path}:{st.st_size}:{st.st_mtime_ns}"
        except Exception:
            return float("nan")

    def run(
        self,
        video,
        force_rate,
        start_time,
        frame_load_cap,
        precision,
        decode_chunk_frames,
        load_audio,
        clean_cache,
        path_override="",
    ):
        return load_video_low_ram(
            video=str(video),
            path_override=str(path_override),
            force_rate=float(force_rate),
            start_time=float(start_time),
            frame_load_cap=int(frame_load_cap),
            precision=str(precision),
            decode_chunk_frames=int(decode_chunk_frames),
            load_audio=bool(load_audio),
            clean_cache=bool(clean_cache),
        )

class AetherScaleVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        gpu_choices, gpu_default = video_gpu_choices()
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "filename_prefix": (
                    "STRING", {"default": "AetherScale/%date:yyyy-MM-dd%/AetherScale", "multiline": False},
                ),
                "container": (["mp4", "mkv", "mov"], {"default": "mp4"}),
                "codec": (
                    ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "prores_proxy", "prores_lt", "prores_standard", "prores_hq", "prores_4444", "prores_4444_xq"],
                    {"default": "h264_nvenc"},
                ),
                "preset": (
                    ["p1", "p2", "p3", "p4", "p5", "p6", "p7"], {"default": "p3"},
                ),
                "nvenc_gpu": (gpu_choices, {"default": gpu_default}),
                "bitrate_mbps": (
                    "INT", {"default": 20, "min": 1, "max": 1000, "step": 1},
                ),
                "pixel_format": (
                    ["yuv420p", "yuv444p"], {"default": "yuv420p"},
                ),
                "save_output": ("BOOLEAN", {"default": True}),
                "chunk_mb": (
                    "INT", {"default": 64, "min": 8, "max": 512, "step": 8},
                ),
                "pipeline_depth": (
                    "INT", {"default": 2, "min": 1, "max": 4, "step": 1},
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_bitrate_kbps": (
                    ["64 kbps", "96 kbps", "128 kbps", "160 kbps", "192 kbps", "256 kbps", "320 kbps", "384 kbps", "448 kbps", "512 kbps"],
                    {"default": "192 kbps"},
                ),
                "nvenc_codec_fallback": ("BOOLEAN", {"default": True}),
                "save_silent_copy": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "metadata_target": (
                    ["sidecar_json", "video_container", "both"],
                    {"default": "sidecar_json"},
                ),
                "seed": (
                    "INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1},
                ),
                "sampler_name": (
                    "STRING", {"default": "", "multiline": False},
                ),
                "scheduler": (
                    "STRING", {"default": "", "multiline": False},
                ),
                "model_name": (
                    "STRING", {"default": "", "multiline": False},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/Output"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "High-throughput IMAGE-to-video encoder with inline preview, generation metadata (seed/sampler/scheduler/model), "
        "audio/silent-copy policy, chunked packing, bounded pipelining, NVENC, and high-depth ProRes/MOV output."
    )

    def run(
        self,
        images,
        frame_rate,
        filename_prefix,
        container,
        codec,
        preset,
        nvenc_gpu,
        bitrate_mbps,
        pixel_format,
        save_output,
        chunk_mb,
        pipeline_depth,
        audio=None,
        audio_bitrate_kbps="192 kbps",
        nvenc_codec_fallback=True,
        save_silent_copy=False,
        save_metadata=True,
        metadata_target="sidecar_json",
        seed=-1,
        sampler_name="",
        scheduler="",
        model_name="",
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        stats = encode_video_batch(
            images,
            frame_rate=float(frame_rate),
            filename_prefix=str(filename_prefix),
            container=str(container),
            codec=str(codec),
            preset=str(preset),
            nvenc_gpu=str(nvenc_gpu),
            bitrate_mbps=int(bitrate_mbps),
            pixel_format=str(pixel_format),
            audio=audio,
            audio_bitrate_kbps=audio_bitrate_kbps,
            nvenc_codec_fallback=bool(nvenc_codec_fallback),
            save_output=bool(save_output),
            chunk_mb=int(chunk_mb),
            pipeline_depth=int(pipeline_depth),
            save_silent_copy=bool(save_silent_copy),
            save_metadata=bool(save_metadata),
            metadata_target=str(metadata_target),
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            unique_id=unique_id,
            generation_seed=int(seed),
            generation_sampler=str(sampler_name),
            generation_scheduler=str(scheduler),
            generation_model=str(model_name),
        )
        result = (str(stats["output_path"]), json.dumps(stats, indent=2))
        preview = stats.get("preview")
        ui = {"gifs": [preview]} if isinstance(preview, dict) else {}
        return {"ui": ui, "result": result}

class AetherScaleDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diagnostics",)
    FUNCTION = "run"
    CATEGORY = "AetherScale/System"

    def run(self, refresh: bool):
        state = RuntimeManager.probe()
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": p.name,
                        "total_vram_gib": round(p.total_memory / (1024**3), 2),
                        "compute_capability": f"{p.major}.{p.minor}",
                    }
                )

        payload = {
            "aetherscale": "0.9.2",
            "runtime_ready": state.ready,
            "runtime_version": state.installed_version,
            "required_runtime_version": state.requested_version,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "devices": devices,
            "effect_cache_entries": VFXBackend.effect_cache_size(),
            "capabilities": VFXBackend.available_capabilities(),
            "neural_rendering": {
                "native_interop": native_nr_runtime_info(),
                "legacy_direct": dlss5_runtime_info(),
            },
            "nodes": [
                "AetherScale • Super Resolution",
                "AetherScale • Restoration",
                "AetherScale • HDR",
                "AetherScale • Motion Analysis",
                "AetherScale • Neural Rendering",
                "AetherScale • Neural VRAM Planner",
                "AetherScale • Runtime",
                "AetherScale • Diagnostics",
                "AetherScale • MFG",
                "AetherScale • MFG Video",
                "AetherScale • Video Loader",
                "AetherScale • Video Combine",
            ],
            "notes": [
                "Super Resolution groups upscale-oriented modes.",
                "Low-RAM Video Loader decodes sequentially into a file-backed IMAGE tensor to avoid whole-video float32 RAM spikes.",
                "Restoration groups same-resolution cleanup modes to avoid node spam.",
                "HDR binding is adaptive because NVIDIA's exposed class names may vary by runtime build.",
                "Neural Rendering now defaults to an in-process native_interop backend with CUDA/D3D12 interop and bounded chunk scheduling.",
                "The NR bridge/caller are declared third-party MIT dependencies; AetherScale owns the adapter, GPU matching, storage, interrupt, and fallback layers.",
                "MFG now defaults to a native D3D12 DLSS Frame Generation backend; the legacy surrogate remains available for fallback/debug.",
                "Legacy surrogate MFG controls remain available for reproducing pre-v0.8 workflows; native DLSSG ignores those artifact-guard controls.",
            ],
        }
        return (json.dumps(payload, indent=2),)




class AetherScaleMotionAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        device_max = max(0, torch.cuda.device_count()-1) if torch.cuda.is_available() else 0
        required = {
            "images": ("IMAGE",),
            "engine": (["auto","torch_lk","nvidia_optical_flow"], {"default":"auto"}),
            "quality": (["mfg_safe","fast","balanced","quality"], {"default":"mfg_safe"}),
            "scene_cut_threshold": ("FLOAT", {"default":0.22,"min":0.01,"max":1.0,"step":0.01}),
            "reset_on_scene_cut": ("BOOLEAN", {"default":True}),
            "cuda_device": ("INT", {"default":0,"min":0,"max":device_max,"step":1}),
            "output_device": (["cpu_safe","same_as_input"], {"default":"cpu_safe"}),
        }
        optional = {
            "motion_mode": (
                ["scene_cuts_only","compact_flow","full_flow"],
                {"default":"compact_flow"},
            ),
            "analysis_long_edge": (
                "INT", {"default":512,"min":128,"max":4096,"step":64},
            ),
            "storage_precision": (
                ["float16","float32"], {"default":"float16"},
            ),
            "preview_frames": (
                "INT", {"default":8,"min":1,"max":32,"step":1},
            ),
        }
        return {"required": required, "optional": optional}
    RETURN_TYPES = ("AETHERSCALE_MOTION","IMAGE","STRING")
    RETURN_NAMES = ("motion","flow_preview","stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/Neural"
    DESCRIPTION = "Temporal motion and scene-cut analysis for future Neural Rendering."
    def run(
        self, images, engine, quality, scene_cut_threshold, reset_on_scene_cut,
        cuda_device, output_device, motion_mode="compact_flow",
        analysis_long_edge=512, storage_precision="float16", preview_frames=8
    ):
        packet = analyze_motion(
            images,
            cuda_device=int(cuda_device),
            engine=("torch_lk" if engine=="auto" else engine),
            quality=quality,
            scene_cut_threshold=float(scene_cut_threshold),
            reset_on_scene_cut=bool(reset_on_scene_cut),
            output_device=output_device,
            motion_mode=motion_mode,
            analysis_long_edge=int(analysis_long_edge),
            storage_precision=storage_precision,
        )
        preview = flow_visualization(packet, max_preview_frames=int(preview_frames))
        stats = {"engine_requested":engine,"engine_resolved":packet.engine,"width":packet.width,"height":packet.height,**packet.metadata}
        return (packet, preview, json.dumps(stats, indent=2))


class AetherScaleNeuralRendering:
    """Backward-compatible DLSS5 NR node.

    Required inputs are frozen to the v0.3.x serialized workflow contract.
    New controls must remain optional or use a new class ID.
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "images": ("IMAGE",),
            "motion": ("AETHERSCALE_MOTION",),
            "style": (
                ["auto", "natural", "cinematic", "material_detail", "default", "3", "4", "5", "6"],
                {"default": "auto"},
            ),
            "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
            "local_tone": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "local_structure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "skin_structure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "reset_on_scene_cut": ("BOOLEAN", {"default": True}),
            "history_frames": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
            "safety_margin_mb": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 128}),
        }
        req.update(_common_runtime_inputs())

        optional = {
            "preset": ("INT", {"default": 3, "min": 0, "max": 3, "step": 1}),
            "auto_mask": ("BOOLEAN", {"default": False}),
            "channel_order": (["auto", "RGBA", "BGRA"], {"default": "auto"}),
            "runtime_path": ("STRING", {"default": "", "multiline": False}),
            "temporal_mode_override": (
                ["legacy_auto", "scene_cut_aware", "temporal_sequence", "still_images"],
                {"default": "legacy_auto"},
            ),
            "output_precision": (
                ["auto", "float16", "float32"], {"default": "auto"},
            ),
            "output_storage": (
                ["auto", "mmap", "ram"], {"default": "auto"},
            ),
            "clean_cache": ("BOOLEAN", {"default": True}),
            "backend": (
                ["native_interop", "carrier", "legacy_direct"], {"default": "native_interop"},
            ),
            "upscale_mode": (
                ["native_1x", "quality_1_5x", "balanced_1_724x", "performance_2x", "ultra_performance_3x"],
                {"default": "native_1x"},
            ),
            "motion_source": (
                ["auto", "connected_motion", "internal_dis", "zero_motion"],
                {"default": "auto"},
            ),
            "carrier_warmup_frames": (
                "INT", {"default": 120, "min": 0, "max": 240, "step": 1},
            ),
            "carrier_scene_cut_threshold": (
                "FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01},
            ),
            "carrier_gpu": (
                carrier_gpu_choices(),
                {"default": "windows_high_performance"},
            ),
            "native_chunk_frames": (
                "INT", {"default": 8, "min": 1, "max": 128, "step": 1},
            ),
            "native_scene_change_threshold": (
                "FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01},
            ),
            "native_fallback": (
                ["carrier", "legacy_direct", "error"], {"default": "carrier"},
            ),
        }
        return {"required": req, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/Neural"
    DESCRIPTION = (
        "DLSS 5 Neural Rendering with an in-process CUDA/D3D12 interop backend by default. "
        "Carrier and legacy direct paths remain available as explicit fallbacks/diagnostics."
    )

    @staticmethod
    def _map_style(style: str) -> str:
        return {
            "auto": "natural",
            "material_detail": "default",
        }.get(style, style)

    def run(
        self,
        images,
        motion,
        style,
        strength,
        local_tone,
        local_structure,
        skin_structure,
        reset_on_scene_cut,
        history_frames,
        safety_margin_mb,
        cuda_device,
        effect_cache,
        cuda_stream,
        memory_policy,
        vram_guard,
        min_free_vram_mb,
        output_device,
        auto_bootstrap,
        preset=3,
        auto_mask=False,
        channel_order="auto",
        runtime_path="",
        temporal_mode_override="legacy_auto",
        output_precision="auto",
        output_storage="auto",
        clean_cache=True,
        backend="carrier",
        upscale_mode="native_1x",
        motion_source="auto",
        carrier_warmup_frames=120,
        carrier_scene_cut_threshold=0.24,
        carrier_gpu="windows_high_performance",
        native_chunk_frames=8,
        native_scene_change_threshold=0.24,
        native_fallback="carrier",
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"Expected IMAGE [T,H,W,C], got {tuple(images.shape)}")

        h, w = int(images.shape[1]), int(images.shape[2])
        if motion is not None and (
            getattr(motion, "width", w) != w or getattr(motion, "height", h) != h
        ):
            raise ValueError(
                f"Motion packet resolution {getattr(motion, 'width', '?')}x"
                f"{getattr(motion, 'height', '?')} does not match input {w}x{h}."
            )

        if temporal_mode_override == "legacy_auto":
            temporal_mode = "scene_cut_aware" if bool(reset_on_scene_cut) else "temporal_sequence"
        else:
            temporal_mode = temporal_mode_override

        effective_min_free_mb = max(int(min_free_vram_mb), int(safety_margin_mb))

        if backend == "native_interop":
            try:
                result, stats = process_native_interop(
                    images,
                    motion=motion,
                    style=self._map_style(str(style)),
                    preset=int(preset),
                    intensity=float(strength),
                    tone=float(local_tone),
                    structure=float(local_structure),
                    skin=float(skin_structure),
                    auto_mask=bool(auto_mask),
                    temporal_mode=temporal_mode,
                    reset_on_scene_cut=bool(reset_on_scene_cut),
                    scene_change_threshold=float(native_scene_change_threshold),
                    gpu_index=int(cuda_device),
                    safety_margin_mb=effective_min_free_mb,
                    vram_guard=vram_guard,
                    output_device=output_device,
                    auto_bootstrap=bool(auto_bootstrap),
                    runtime_path=runtime_path,
                    output_precision=output_precision,
                    output_storage=output_storage,
                    clean_cache=bool(clean_cache),
                    upscale_mode=upscale_mode,
                    chunk_frames=int(native_chunk_frames),
                    channel_order=channel_order,
                )
            except NativeNRInteropError as exc:
                if native_fallback == "error":
                    raise
                print(
                    f"[AetherScale] Native NR interop unavailable; explicit fallback "
                    f"{native_fallback}: {exc}",
                    flush=True,
                )
                if native_fallback == "carrier":
                    result, stats = process_carrier(
                        images,
                        motion=motion,
                        style=str(style),
                        preset=int(preset),
                        intensity=float(strength),
                        tone=float(local_tone),
                        structure=float(local_structure),
                        skin=float(skin_structure),
                        auto_mask=bool(auto_mask),
                        upscale_mode=upscale_mode,
                        warmup_frames=int(carrier_warmup_frames),
                        scene_cut_threshold=float(carrier_scene_cut_threshold),
                        motion_source=motion_source,
                        auto_bootstrap=bool(auto_bootstrap),
                        output_precision=output_precision,
                        output_storage=output_storage,
                        clean_cache=bool(clean_cache),
                        carrier_gpu=carrier_gpu,
                    )
                else:
                    result, stats = process_dlss5(
                        images,
                        style=self._map_style(str(style)),
                        preset=int(preset),
                        intensity=float(strength),
                        tone=float(local_tone),
                        structure=float(local_structure),
                        skin=float(skin_structure),
                        auto_mask=bool(auto_mask),
                        temporal_mode=temporal_mode,
                        channel_order=channel_order,
                        gpu_index=int(cuda_device),
                        vram_guard=vram_guard,
                        min_free_vram_mb=effective_min_free_mb,
                        output_device=output_device,
                        auto_bootstrap=bool(auto_bootstrap),
                        runtime_path=runtime_path,
                        motion=motion,
                        output_precision=output_precision,
                        output_storage=output_storage,
                        clean_cache=bool(clean_cache),
                    )
                stats["native_interop_fallback_reason"] = f"{type(exc).__name__}: {exc}"
                stats["native_interop_fallback_backend"] = native_fallback
        elif backend == "carrier":
            result, stats = process_carrier(
                images,
                motion=motion,
                style=str(style),
                preset=int(preset),
                intensity=float(strength),
                tone=float(local_tone),
                structure=float(local_structure),
                skin=float(skin_structure),
                auto_mask=bool(auto_mask),
                upscale_mode=upscale_mode,
                warmup_frames=int(carrier_warmup_frames),
                scene_cut_threshold=float(carrier_scene_cut_threshold),
                motion_source=motion_source,
                auto_bootstrap=bool(auto_bootstrap),
                output_precision=output_precision,
                output_storage=output_storage,
                clean_cache=bool(clean_cache),
                carrier_gpu=carrier_gpu,
            )
        else:
            result, stats = process_dlss5(
                images,
                style=self._map_style(str(style)),
                preset=int(preset),
                intensity=float(strength),
                tone=float(local_tone),
                structure=float(local_structure),
                skin=float(skin_structure),
                auto_mask=bool(auto_mask),
                temporal_mode=temporal_mode,
                channel_order=channel_order,
                gpu_index=int(cuda_device),
                vram_guard=vram_guard,
                min_free_vram_mb=effective_min_free_mb,
                output_device=output_device,
                auto_bootstrap=bool(auto_bootstrap),
                runtime_path=runtime_path,
                motion=motion,
                output_precision=output_precision,
                output_storage=output_storage,
                clean_cache=bool(clean_cache),
            )

        stats["compatibility_contract"] = "AetherScaleNeuralRendering/v0.3-required-schema"
        stats["legacy_settings"] = {
            "style": style,
            "strength": float(strength),
            "local_tone": float(local_tone),
            "local_structure": float(local_structure),
            "skin_structure": float(skin_structure),
            "reset_on_scene_cut": bool(reset_on_scene_cut),
            "history_frames": int(history_frames),
            "safety_margin_mb": int(safety_margin_mb),
            "effect_cache": effect_cache,
            "cuda_stream": cuda_stream,
            "memory_policy": memory_policy,
        }
        stats["effective_min_free_vram_mb"] = effective_min_free_mb
        stats["native_settings"] = {
            "chunk_frames": int(native_chunk_frames),
            "scene_change_threshold": float(native_scene_change_threshold),
            "fallback": str(native_fallback),
        }
        return (result, json.dumps(stats, indent=2))


class AetherScaleNeuralPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default":3840,"min":64,"max":16384,"step":8}),
            "height": ("INT", {"default":2160,"min":64,"max":16384,"step":8}),
            "history_frames": ("INT", {"default":2,"min":0,"max":8,"step":1}),
            "safety_margin_mb": ("INT", {"default":2048,"min":0,"max":16384,"step":128}),
            "measured_context_mb": ("INT", {"default":0,"min":0,"max":16384,"step":64}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plan",)
    FUNCTION = "run"
    CATEGORY = "AetherScale/Neural"
    def run(self, width, height, history_frames, safety_margin_mb, measured_context_mb):
        payload = neural_vram_plan(int(width),int(height),history_frames=int(history_frames),safety_margin_mb=int(safety_margin_mb),measured_context_mb=int(measured_context_mb))
        payload["dlssnr"] = {"native_interop": native_nr_runtime_info(), "legacy_direct": dlss5_runtime_info()}
        return (json.dumps(payload, indent=2),)


NODE_CLASS_MAPPINGS = {
    "AetherScaleSuperResolution": AetherScaleSuperResolution,
    "AetherScaleRestoration": AetherScaleRestoration,
    "AetherScaleHDR": AetherScaleHDR,
    "AetherScaleMotionAnalysis": AetherScaleMotionAnalysis,
    "AetherScaleNeuralRendering": AetherScaleNeuralRendering,
    "AetherScaleNeuralPlanner": AetherScaleNeuralPlanner,
    "AetherScaleRuntime": AetherScaleRuntime,
    "AetherScaleDiagnostics": AetherScaleDiagnostics,
    "AetherScaleMFGLab": AetherScaleMFGLab,
    "AetherScaleMFGVideo": AetherScaleMFGVideo,
    "AetherScaleVideoLoader": AetherScaleVideoLoader,
    "AetherScaleVideoCombine": AetherScaleVideoCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AetherScaleSuperResolution": "AetherScale • Super Resolution",
    "AetherScaleRestoration": "AetherScale • Restoration",
    "AetherScaleHDR": "AetherScale • HDR",
    "AetherScaleMotionAnalysis": "AetherScale • Motion Analysis",
    "AetherScaleNeuralRendering": "AetherScale • Neural Rendering",
    "AetherScaleNeuralPlanner": "AetherScale • Neural VRAM Planner",
    "AetherScaleRuntime": "AetherScale • Runtime",
    "AetherScaleDiagnostics": "AetherScale • Diagnostics",
    "AetherScaleMFGLab": "AetherScale • MFG",
    "AetherScaleMFGVideo": "AetherScale • MFG Video",
    "AetherScaleVideoLoader": "AetherScale • Video Loader",
    "AetherScaleVideoCombine": "AetherScale • Video Combine",
}


# Every AetherScale node reports start/finish/error timing in the console.
# Long frame-processing backends additionally emit detailed progress bars with
# frame/pair throughput and ETA through backend.progress.ConsoleProgress.
for _node_id, _node_cls in NODE_CLASS_MAPPINGS.items():
    _display = NODE_DISPLAY_NAME_MAPPINGS.get(_node_id, _node_id)
    _fn_name = getattr(_node_cls, "FUNCTION", "run")
    _fn = getattr(_node_cls, _fn_name, None)
    if callable(_fn) and not getattr(_fn, "_aetherscale_console_wrapped", False):
        setattr(_node_cls, _fn_name, console_node(_display)(_fn))
