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
        return {"required": req}

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
        )
        return (result, json.dumps(stats, indent=2))


class AetherScaleRuntime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (
                    ["status", "install_or_update", "repair", "clear_effect_cache", "install_dlss5_bridge", "shutdown_dlss5", "clear_runtime"],
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
            "neural_rendering": dlss5_runtime_info(),
        }
        return (json.dumps(info, indent=2),)


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
            "aetherscale": "0.5.4",
            "runtime_ready": state.ready,
            "runtime_version": state.installed_version,
            "required_runtime_version": state.requested_version,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "devices": devices,
            "effect_cache_entries": VFXBackend.effect_cache_size(),
            "capabilities": VFXBackend.available_capabilities(),
            "neural_rendering": dlss5_runtime_info(),
            "nodes": [
                "AetherScale • Super Resolution",
                "AetherScale • Restoration",
                "AetherScale • HDR",
                "AetherScale • Motion Analysis",
                "AetherScale • Neural Rendering",
                "AetherScale • Neural VRAM Planner",
                "AetherScale • Runtime",
                "AetherScale • Diagnostics",
            ],
            "notes": [
                "Super Resolution groups upscale-oriented modes.",
                "Restoration groups same-resolution cleanup modes to avoid node spam.",
                "HDR binding is adaptive because NVIDIA's exposed class names may vary by runtime build.",
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
            "quality": (["balanced","quality","fast"], {"default":"balanced"}),
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
                ["carrier", "legacy_direct"], {"default": "carrier"},
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
        }
        return {"required": req, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stats")
    FUNCTION = "run"
    CATEGORY = "AetherScale/Neural"
    DESCRIPTION = (
        "DLSS 5 Neural Rendering using carrier DLSS + temporal motion guides. "
        "Legacy direct feature-18 path remains available only for diagnostics."
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

        if backend == "carrier":
            # Primary architecture: normal DLSS carrier + motion guides + RenoDX
            # Neural Rendering injection. This avoids naked CreateFeature(18).
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
            # Legacy diagnostic backend only. Retained so existing experiments
            # can still be reproduced, but it is no longer the default path.
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
        payload["dlssnr"] = dlss5_runtime_info()
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
}
