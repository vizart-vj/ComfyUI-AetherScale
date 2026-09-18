
from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .neural import MotionPacket, analyze_motion
from .storage import allocate_cpu_tensor, resolve_dtype, estimate_bytes, sync_file_backed_tensor
from .progress import ConsoleProgress
from .dlssg_native import (
    NativeDLSSGError, generate_native_mfg_images, probe_native_capabilities, stream_native_mfg,
)

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows import safety
    winreg = None


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        cp = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,driver_version,memory.total,pci.bus_id,uuid',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if cp.returncode != 0:
            return out
        for line in cp.stdout.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                out.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'driver_version': parts[2],
                    'memory_mb': int(float(parts[3])),
                    'pci_bus_id': parts[4],
                    'uuid': parts[5] if len(parts) >= 6 else '',
                })
    except Exception:
        pass
    return out


def _parse_driver(version: str) -> tuple[int, ...]:
    nums = re.findall(r'\d+', str(version))
    return tuple(int(x) for x in nums[:3]) if nums else (0,)


def _windows_build() -> int | None:
    if os.name != 'nt':
        return None
    try:
        v = sys_getwindowsversion()
        return int(v.build)
    except Exception:
        pass
    try:
        return int(platform.version().split('.')[-1])
    except Exception:
        return None


def sys_getwindowsversion():
    fn = getattr(platform, 'win32_ver', None)
    if fn is None:
        raise RuntimeError('win32 version not available')
    # We need the build; platform module does not expose it directly in all envs,
    # so use sys if available.
    import sys
    return sys.getwindowsversion()  # type: ignore[attr-defined]


def _hags_state() -> dict[str, Any]:
    info: dict[str, Any] = {'supported': os.name == 'nt', 'raw': None, 'enabled': None}
    if os.name != 'nt' or winreg is None:
        return info
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r'SYSTEM\CurrentControlSet\Control\GraphicsDrivers',
        )
        value, _ = winreg.QueryValueEx(key, 'HwSchMode')
        info['raw'] = int(value)
        if int(value) == 2:
            info['enabled'] = True
        elif int(value) == 1:
            info['enabled'] = False
        else:
            info['enabled'] = None
    except FileNotFoundError:
        info['reason'] = 'registry_value_missing'
    except Exception as exc:
        info['reason'] = f'{type(exc).__name__}: {exc}'
    return info


def _gpu_generation(name: str) -> int:
    up = str(name).upper()
    m = re.search(r'RTX\s+(\d{2})', up)
    if m:
        return int(m.group(1))
    return 0


def probe_mfg_environment(preferred_gpu: Optional[int] = None) -> dict[str, Any]:
    gpus = _nvidia_smi_gpus()
    selected = None
    if preferred_gpu is not None:
        for gpu in gpus:
            if gpu['index'] == preferred_gpu:
                selected = gpu
                break
    if selected is None and gpus:
        selected = max(gpus, key=lambda g: (_gpu_generation(g['name']), int(g['memory_mb'])))

    windows_build = _windows_build()
    hags = _hags_state()
    selected_name = str(selected['name']) if selected else ''
    driver = _parse_driver(selected['driver_version']) if selected else (0,)
    gen = _gpu_generation(selected_name)

    # Public docs: DLSS-G requires Win10 20H1+ and HAGS. Dynamic/MFG class features
    # on Blackwell are treated as likely on RTX 50 with driver >= 595.41.
    os_ok = windows_build is not None and windows_build >= 19041
    driver_ok = driver >= (595, 41)
    dlssg_likely = gen >= 40
    mfg_likely = gen >= 50

    reasons: list[str] = []
    if not os_ok:
        reasons.append('Windows build below 19041 or unavailable')
    if hags.get('enabled') is False:
        reasons.append('Hardware-accelerated GPU scheduling (HAGS) is disabled')
    elif hags.get('enabled') is None:
        reasons.append('HAGS state could not be verified')
    if not driver_ok:
        reasons.append('Driver below 595.41 or unavailable')
    if not selected:
        reasons.append('No NVIDIA GPU detected by nvidia-smi')
    elif not dlssg_likely:
        reasons.append('Selected GPU is below RTX 40 class')

    return {
        'probe_kind': 'streamline_mfg_lab',
        'native_host_available': True,
        'selected_gpu': selected,
        'detected_gpus': gpus,
        'windows_build': windows_build,
        'hags': hags,
        'driver_ok_for_modern_streamline': driver_ok,
        'dlssg_likely_supported': bool(selected and dlssg_likely and os_ok),
        'mfg_likely_supported': bool(selected and mfg_likely and os_ok and driver_ok),
        'reasons': reasons,
        'notes': [
            'AetherScale 0.8 includes a native D3D12 DLSSG worker backend plus the legacy surrogate fallback.',
            'The native backend exchanges RGBA frames and FP16 motion guides directly with the worker.',
            'Use probe_only to inspect the runtime-supported native multiplier before a long job.',
        ],
    }


def _warp_nchw(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if img.ndim != 4 or flow.ndim != 4:
        raise ValueError('Expected img [1,C,H,W] and flow [1,2,H,W].')
    _, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=torch.float32),
        torch.arange(w, device=img.device, dtype=torch.float32),
        indexing='ij',
    )
    gx = xx[None] + flow[:, 0]
    gy = yy[None] + flow[:, 1]
    gx = 2 * gx / max(w - 1, 1) - 1
    gy = 2 * gy / max(h - 1, 1) - 1
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )


def _upsample_flow(flow_hwc: torch.Tensor, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    x = flow_hwc.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    src_h, src_w = int(x.shape[-2]), int(x.shape[-1])
    if (src_h, src_w) != (target_h, target_w):
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
        x[:, 0] *= float(target_w) / float(max(src_w, 1))
        x[:, 1] *= float(target_h) / float(max(src_h, 1))
    return x.contiguous()


def _upsample_conf(conf_hwc: torch.Tensor, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    x = conf_hwc.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    if tuple(x.shape[-2:]) != (target_h, target_w):
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
    return x.clamp(0.0, 1.0).contiguous()


def _linear_tween(prev: torch.Tensor, curr: torch.Tensor, t: float) -> torch.Tensor:
    return ((1.0 - t) * prev + t * curr).clamp(0.0, 1.0)


def _scene_cut_fill(prev: torch.Tensor, curr: torch.Tensor, t: float, strategy: str) -> torch.Tensor:
    if strategy == 'repeat_current':
        return curr
    if strategy == 'linear_blend':
        return _linear_tween(prev, curr, t)
    return prev


def _luma(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).contiguous()


def _high_frequency_mask(x: torch.Tensor) -> torch.Tensor:
    lum = _luma(x)
    low = F.avg_pool2d(lum, kernel_size=3, stride=1, padding=1)
    hf = (lum - low).abs()
    return ((hf - 0.025) / 0.12).clamp(0.0, 1.0)


def _emissive_mask(x: torch.Tensor) -> torch.Tensor:
    lum = _luma(x)
    peak = x.amax(dim=1, keepdim=True)
    mean = x.mean(dim=1, keepdim=True)
    glow = ((peak - 0.55) / 0.35).clamp(0.0, 1.0)
    color_sep = ((peak - mean - 0.08) / 0.22).clamp(0.0, 1.0)
    bright = ((lum - 0.25) / 0.5).clamp(0.0, 1.0)
    return (glow * 0.5 + color_sep * 0.35 + bright * 0.15).clamp(0.0, 1.0)


def _dilate_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0.0, 1.0)
    kernel = radius * 2 + 1
    return F.max_pool2d(mask.clamp(0.0, 1.0), kernel_size=kernel, stride=1, padding=radius)


def _closest_source(prev: torch.Tensor, curr: torch.Tensor, t: float) -> torch.Tensor:
    return prev if float(t) < 0.5 else curr


def _build_guard_context(
    prev: torch.Tensor,
    curr: torch.Tensor,
    flow: torch.Tensor,
    conf: torch.Tensor,
    *,
    artifact_guard: float,
    emissive_protection: float,
    thin_detail_protection: float,
    mv_confidence_threshold: float,
) -> dict[str, torch.Tensor | float]:
    wp_full = _warp_nchw(prev, flow)
    wc_full = _warp_nchw(curr, -flow)
    residual_curr = (wp_full - curr).abs().mean(dim=1, keepdim=True)
    residual_prev = (wc_full - prev).abs().mean(dim=1, keepdim=True)
    residual = torch.maximum(residual_curr, residual_prev)
    residual_risk = ((residual - 0.025) / 0.16).clamp(0.0, 1.0)

    conf_risk = (1.0 - conf.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    detail_risk = torch.maximum(_high_frequency_mask(prev), _high_frequency_mask(curr))
    emissive_risk = torch.maximum(_emissive_mask(prev), _emissive_mask(curr))

    combined = torch.maximum(residual_risk, conf_risk * 0.75)
    combined = torch.maximum(combined, detail_risk * float(thin_detail_protection))
    combined = torch.maximum(combined, emissive_risk * float(emissive_protection))
    combined = _dilate_mask(combined, radius=1)

    stability = 1.0 - combined * float(artifact_guard)
    stability = stability.clamp(0.0, 1.0)
    unstable = (stability < float(mv_confidence_threshold)).to(torch.float32)
    unstable = _dilate_mask(unstable, radius=1)

    return {
        'stability': stability,
        'unstable': unstable,
        'residual_risk': residual_risk,
        'detail_risk': detail_risk,
        'emissive_risk': emissive_risk,
    }


def _mv_surrogate(
    prev: torch.Tensor,
    curr: torch.Tensor,
    flow: torch.Tensor,
    conf: torch.Tensor,
    t: float,
    *,
    guard_context: dict[str, torch.Tensor | float] | None = None,
    fallback_mode: str = 'linear_blend',
    synthesis_mode: str = 'continuous_temporal',
) -> torch.Tensor:
    # flow is current->previous.  For an intermediate time t, both source
    # frames are warped INTO the intermediate temporal position:
    #   prev -> t      :  t * flow(current->previous)
    #   curr -> t      : -(1-t) * flow(current->previous)
    # This keeps generated frames at true sub-frame times instead of merely
    # repeating source frames.
    wp = _warp_nchw(prev, flow * float(t))
    wc = _warp_nchw(curr, -flow * float(1.0 - t))
    base = _linear_tween(prev, curr, t)
    alpha = conf.clamp(0.0, 1.0) * 0.65 + 0.15
    mixed = ((1.0 - t) * wp + t * wc).clamp(0.0, 1.0)

    if guard_context is None:
        return (mixed * alpha + base * (1.0 - alpha)).clamp(0.0, 1.0)

    stability = guard_context['stability']  # type: ignore[index]
    unstable = guard_context['unstable']  # type: ignore[index]

    if synthesis_mode == 'legacy_guarded':
        # v0.6.3-v0.7.4 behaviour retained only for reproducibility.  The
        # closest_source option can snap protected pixels to an endpoint and is
        # therefore unsuitable for real slow-motion interpolation.
        legacy_alpha = alpha * stability
        guided = (mixed * legacy_alpha + base * (1.0 - legacy_alpha)).clamp(0.0, 1.0)
        if fallback_mode == 'closest_source':
            fallback = _closest_source(prev, curr, t)
        else:
            fallback = base
        return (guided * (1.0 - unstable) + fallback * unstable).clamp(0.0, 1.0)

    # Continuous temporal synthesis (default): never switch between temporal
    # branches at t=0.5.  v0.7.5 still used `wp if t < .5 else wc` inside the
    # guard fallback.  That removed endpoint freezes, but introduced a new hard
    # midpoint branch switch: thin bright details could jump between two
    # slightly different warp solutions from one generated frame to the next.
    #
    # The packet flow is current->previous and is therefore natively suited to
    # reconstructing intermediate target times from PREV with `t * flow`.
    # `wc` uses an approximate inverse (-flow) on the wrong spatial lattice, so
    # it is useful as a secondary estimate but must never become a binary source
    # selector for protected pixels.
    guided_alpha = alpha * stability
    guided = (mixed * guided_alpha + base * (1.0 - guided_alpha)).clamp(0.0, 1.0)

    # Detect per-time disagreement between the two candidate warps.  When they
    # disagree, dual-source blending is precisely what creates doubled torches,
    # LEDs, sparks and other sub-pixel/high-contrast details.
    branch_delta = (wp - wc).abs().mean(dim=1, keepdim=True)
    branch_disagreement = ((branch_delta - 0.018) / 0.12).clamp(0.0, 1.0)

    # Guard strength is continuous in both space and time.  High-risk regions
    # lock to the native backward-flow branch (`wp`) for the ENTIRE source pair,
    # rather than switching branches at the midpoint.  This preserves a single
    # continuous trajectory for emissive/thin details.
    guard_amount = torch.maximum((1.0 - stability).clamp(0.0, 1.0), branch_disagreement)

    # At very low confidence, soften the source-locked result slightly toward
    # the RGB temporal baseline, but keep the motion branch dominant so the
    # detail does not split into two independently moving copies.
    source_conf = (0.70 + 0.30 * conf.clamp(0.0, 1.0)).clamp(0.70, 1.0)
    source_locked = (wp * source_conf + base * (1.0 - source_conf)).clamp(0.0, 1.0)

    return (guided * (1.0 - guard_amount) + source_locked * guard_amount).clamp(0.0, 1.0)


def generate_mfg_lab(
    images: torch.Tensor,
    *,
    motion: Optional[MotionPacket],
    mode: str,
    motion_source: str,
    multiplier: int,
    surrogate_method: str,
    scene_cut_strategy: str,
    cuda_device: int,
    output_device: str,
    output_precision: str = "auto",
    output_storage: str = "auto",
    clean_cache: bool = True,
    artifact_guard: float = 1.0,
    emissive_protection: float = 0.85,
    thin_detail_protection: float = 0.75,
    mv_confidence_threshold: float = 0.45,
    fallback_mode: str = "closest_source",
    synthesis_mode: str = "continuous_temporal",
    source_frame_rate: float = 24.0,
    native_auto_bootstrap: bool = True,
    native_guide_source: str = "internal_dis",
    native_scene_cut_threshold: float = 0.24,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(f"Expected IMAGE [T,H,W,C], got {tuple(images.shape)}")
    if multiplier < 1:
        raise ValueError("multiplier must be >= 1")

    env = probe_mfg_environment(preferred_gpu=int(cuda_device))
    if mode == "probe_only":
        native = probe_native_capabilities(auto_bootstrap=bool(native_auto_bootstrap), force=True)
        stats = {
            "mode": mode,
            "environment": env,
            "native_dlssg": native,
            "backend_resolved": "probe_only",
            "output_frames": int(images.shape[0]),
            "lab_result": "probe_only_passthrough",
        }
        return images, stats

    if mode == "native_dlssg":
        out, native_stats = generate_native_mfg_images(
            images,
            source_frame_rate=float(source_frame_rate),
            multiplier=int(multiplier),
            guide_source=str(native_guide_source),
            motion=motion,
            scene_cut_threshold=float(native_scene_cut_threshold),
            auto_bootstrap=bool(native_auto_bootstrap),
            output_precision=str(output_precision),
            output_storage=str(output_storage),
            clean_cache=bool(clean_cache),
        )
        native_stats["mode"] = mode
        native_stats["backend_resolved"] = "native_dlssg"
        native_stats["environment"] = env
        native_stats["legacy_surrogate_controls_ignored"] = {
            "artifact_guard": float(artifact_guard),
            "emissive_protection": float(emissive_protection),
            "thin_detail_protection": float(thin_detail_protection),
            "mv_confidence_threshold": float(mv_confidence_threshold),
            "fallback_mode": str(fallback_mode),
            "synthesis_mode": str(synthesis_mode),
            "surrogate_method": str(surrogate_method),
        }
        if output_device == "same_as_input" and images.device.type != "cpu":
            if out.numel() * out.element_size() <= 512 * 1024 * 1024:
                out = out.to(images.device)
                native_stats["output_device_resolved"] = str(images.device)
            else:
                native_stats["output_device_resolved"] = "cpu_safe_forced_large_output"
        else:
            native_stats["output_device_resolved"] = "cpu"
        return out, native_stats
    stats: dict[str, Any] = {
        "mode": mode,
        "multiplier": int(multiplier),
        "surrogate_method": surrogate_method,
        "scene_cut_strategy": scene_cut_strategy,
        "artifact_guard": float(artifact_guard),
        "emissive_protection": float(emissive_protection),
        "thin_detail_protection": float(thin_detail_protection),
        "mv_confidence_threshold": float(mv_confidence_threshold),
        "fallback_mode": str(fallback_mode),
        "synthesis_mode": str(synthesis_mode),
        "continuous_temporal_branch_policy": "source_locked_prev_no_midpoint_switch",
        "temporal_positions": [round(i / float(multiplier), 6) for i in range(1, int(multiplier))],
        "environment": env,
        "native_streamline_host": "not_included_in_this_build",
    }

    n, h, w, c = [int(x) for x in images.shape]
    if n < 2 or multiplier == 1:
        stats["output_frames"] = n
        stats["lab_result"] = "passthrough"
        return images, stats

    packet = motion
    resolved_motion_source = motion_source
    if packet is None or packet.flow.numel() == 0:
        if motion_source == "connected_motion":
            resolved_motion_source = "none_available"
            packet = None
        elif motion_source == "rgb_only":
            resolved_motion_source = "rgb_only"
            packet = None
        else:
            if torch.cuda.is_available():
                resolved_motion_source = "internal_compact"
                packet = analyze_motion(
                    images,
                    cuda_device=int(cuda_device),
                    engine="torch_lk",
                    quality="mfg_safe",
                    scene_cut_threshold=0.24,
                    reset_on_scene_cut=True,
                    output_device="cpu_safe",
                    motion_mode="compact_flow",
                    analysis_long_edge=512,
                    storage_precision="float16",
                )
            else:
                resolved_motion_source = "rgb_only_fallback_no_cuda"
                packet = None
    stats["motion_source_resolved"] = resolved_motion_source

    # Exact final frame count for N source frames and Kx presentation rate:
    # (N-1)*K + 1.  Allocate once, then stream directly into the backing store.
    out_frames = (n - 1) * int(multiplier) + 1
    out_shape = (out_frames, h, w, c)
    out_dtype = resolve_dtype(
        requested=str(output_precision),
        shape=out_shape,
        input_dtype=images.dtype,
        auto_fp16_threshold_mb=512,
    )
    estimated_bytes = estimate_bytes(out_shape, out_dtype)
    max_image_gib = float(os.environ.get("AETHERSCALE_MFG_IMAGE_MAX_GIB", "64"))
    if estimated_bytes > int(max_image_gib * (1024 ** 3)):
        raise RuntimeError(
            "MFG IMAGE output is too large to materialize safely as a ComfyUI IMAGE batch. "
            f"This {mode} request would require {estimated_bytes / (1024 ** 3):.2f} GiB "
            f"({out_frames} frames at {w}x{h}, {str(out_dtype).replace('torch.', '')}). "
            "Use AetherScale • MFG Video after spatial/HDR processing so generated frames are "
            "streamed directly to the encoder. This guard applies to both native and surrogate MFG. "
            "For a deliberate huge IMAGE materialization, raise AETHERSCALE_MFG_IMAGE_MAX_GIB."
        )
    out, storage = allocate_cpu_tensor(
        out_shape,
        dtype=out_dtype,
        storage_mode=str(output_storage),
        prefix="mfg",
        mmap_threshold_mb=512,
        clean_cache=bool(clean_cache),
    )
    console_progress = ConsoleProgress("MFG / surrogate", out_frames, unit="frame")
    stats.update({
        "input_frames": n,
        "output_frames": out_frames,
        "output_dtype": str(out_dtype).replace("torch.", ""),
        "output_storage_backend": storage.backend,
        "output_storage_bytes": int(storage.bytes),
        "output_storage_gib": round(storage.bytes / (1024 ** 3), 3),
        "output_storage_path": storage.path,
        "output_storage_fallback_reason": storage.fallback_reason,
        "system_commit_available_gib_at_allocation": (
            round(storage.commit_available_bytes / 1024**3, 3)
            if storage.commit_available_bytes is not None else None
        ),
        "spill_disk_free_gib_at_allocation": (
            round(storage.disk_free_bytes / 1024**3, 3)
            if storage.disk_free_bytes is not None else None
        ),
        "streaming_output_writer": True,
        "uses_torch_stack": False,
        "clean_cache": bool(clean_cache),
    })

    def write_frame(index: int, frame_hwc: torch.Tensor) -> None:
        # One frame at a time. Never concatenate or materialize the full output.
        src = frame_hwc.detach()
        if src.device.type != "cpu":
            src = src.to("cpu", non_blocking=False)
        if src.dtype != out_dtype:
            src = src.to(dtype=out_dtype)
        out[index].copy_(src, non_blocking=False)
        sync_file_backed_tensor(out, bytes_written=src.numel() * src.element_size())
        console_progress.update(1)

    # RGB-only fallback still uses the same streaming writer, avoiding a giant stack.
    if packet is None or packet.flow.numel() == 0:
        oi = 0
        for i in range(n - 1):
            prev_hwc = images[i]
            curr_hwc = images[i + 1]
            write_frame(oi, prev_hwc)
            oi += 1
            for sub in range(1, multiplier):
                t = sub / float(multiplier)
                if surrogate_method == "frame_repeat":
                    frame = prev_hwc
                else:
                    frame = _linear_tween(prev_hwc, curr_hwc, t)
                write_frame(oi, frame)
                oi += 1
        write_frame(oi, images[-1])
        stats["lab_result"] = "rgb_only_fallback"
        stats["inserted_frames"] = out_frames - n
        sync_file_backed_tensor(out, force=True)
        console_progress.close(status="done")
        return out, stats

    work_device = (
        torch.device(f"cuda:{int(cuda_device)}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    oi = 0
    inserted = 0
    cut_pairs = 0
    guarded_pairs = 0
    unstable_ratio_sum = 0.0
    emissive_ratio_sum = 0.0
    detail_ratio_sum = 0.0

    for i in range(n - 1):
        prev_hwc = images[i]
        curr_hwc = images[i + 1]
        write_frame(oi, prev_hwc)
        oi += 1

        cut = bool(packet.scene_cuts[i].item()) if packet.scene_cuts.numel() else False
        if cut:
            cut_pairs += 1

        prev = prev_hwc.to(device=work_device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()
        curr = curr_hwc.to(device=work_device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()

        flow = conf = guard_context = None
        if not cut:
            flow = _upsample_flow(packet.flow[i].to("cpu"), h, w, work_device)
            if packet.confidence.numel():
                conf = _upsample_conf(packet.confidence[i].to("cpu"), h, w, work_device)
            else:
                conf = torch.ones((1, 1, h, w), device=work_device, dtype=torch.float32)
            guard_context = _build_guard_context(
                prev,
                curr,
                flow,
                conf,
                artifact_guard=float(artifact_guard),
                emissive_protection=float(emissive_protection),
                thin_detail_protection=float(thin_detail_protection),
                mv_confidence_threshold=float(mv_confidence_threshold),
            )
            guarded_pairs += 1
            unstable_ratio_sum += float(guard_context['unstable'].mean().item())
            emissive_ratio_sum += float(guard_context['emissive_risk'].mean().item())
            detail_ratio_sum += float(guard_context['detail_risk'].mean().item())

        for sub in range(1, multiplier):
            t = sub / float(multiplier)
            if cut:
                frame_nchw = _scene_cut_fill(prev, curr, t, scene_cut_strategy)
            elif surrogate_method == "frame_repeat":
                frame_nchw = prev
            elif surrogate_method == "linear_blend":
                frame_nchw = _linear_tween(prev, curr, t)
            else:
                assert flow is not None and conf is not None
                frame_nchw = _mv_surrogate(
                    prev, curr, flow, conf, t,
                    guard_context=guard_context,
                    fallback_mode=str(fallback_mode),
                    synthesis_mode=str(synthesis_mode),
                )

            frame_hwc = frame_nchw[0].permute(1, 2, 0)
            write_frame(oi, frame_hwc)
            oi += 1
            inserted += 1

        del prev, curr
        if flow is not None:
            del flow
        if conf is not None:
            del conf
        if guard_context is not None:
            del guard_context

    write_frame(oi, images[-1])

    stats.update({
        "lab_result": "motion_vector_surrogate",
        "inserted_frames": int(inserted),
        "scene_cut_pairs": int(cut_pairs),
        "guarded_pairs": int(guarded_pairs),
        "avg_unstable_ratio": round(unstable_ratio_sum / max(guarded_pairs, 1), 6),
        "avg_emissive_risk": round(emissive_ratio_sum / max(guarded_pairs, 1), 6),
        "avg_detail_risk": round(detail_ratio_sum / max(guarded_pairs, 1), 6),
        "motion_packet_direction": getattr(packet, "metadata", {}).get("direction", "unknown"),
        "motion_packet_flow_resolution": [
            int(getattr(packet, "metadata", {}).get("flow_width", 0)),
            int(getattr(packet, "metadata", {}).get("flow_height", 0)),
        ],
    })

    # Keep long outputs CPU-backed. Moving a multi-GB batch back to CUDA at once
    # would defeat the streaming architecture and can cause immediate OOM.
    if output_device == "same_as_input" and images.device.type != "cpu":
        if storage.bytes <= 512 * 1024 * 1024:
            out = out.to(images.device)
            stats["output_device_resolved"] = str(images.device)
        else:
            stats["output_device_resolved"] = "cpu_safe_forced_large_output"
    else:
        stats["output_device_resolved"] = "cpu"
    sync_file_backed_tensor(out, force=True)
    console_progress.close(status="done")
    return out, stats


def _find_ffmpeg() -> str:
    import shutil

    explicit = os.environ.get("AETHERSCALE_FFMPEG") or os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if explicit and os.path.isfile(explicit):
        return explicit

    path = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    if path:
        return path

    # If VideoHelperSuite is installed, reuse the exact ffmpeg binary it already
    # selected. Import lazily so AetherScale has no hard dependency on VHS.
    try:
        from videohelpersuite.utils import ffmpeg_path as vhs_ffmpeg  # type: ignore
        if vhs_ffmpeg and os.path.isfile(vhs_ffmpeg):
            return str(vhs_ffmpeg)
    except Exception:
        pass

    # Common portable-ComfyUI layouts / custom-node bundles.
    here = os.path.abspath(os.path.dirname(__file__))
    custom_nodes = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.path.join(custom_nodes, "ComfyUI-VideoHelperSuite", "videohelpersuite", "ffmpeg.exe"),
        os.path.join(custom_nodes, "ComfyUI-VideoHelperSuite", "ffmpeg.exe"),
        os.path.join(custom_nodes, "VideoHelperSuite", "videohelpersuite", "ffmpeg.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        "FFmpeg was not found. Install/enable VideoHelperSuite, put ffmpeg in PATH, "
        "or set AETHERSCALE_FFMPEG / VHS_FORCE_FFMPEG_PATH."
    )


def _resolve_video_output_path(filename_prefix: str, extension: str) -> tuple[str, str]:
    """Return absolute output path and user-facing subfolder/name."""
    import datetime

    extension = extension.lstrip(".")
    prefix = str(filename_prefix)
    # Match the date token style commonly used in VideoHelperSuite workflows.
    now = datetime.datetime.now()
    token_map = {
        "yyyy": f"{now.year:04d}", "MM": f"{now.month:02d}", "dd": f"{now.day:02d}",
        "HH": f"{now.hour:02d}", "mm": f"{now.minute:02d}", "ss": f"{now.second:02d}",
    }
    def repl(match):
        fmt = match.group(1)
        for key in ("yyyy", "MM", "dd", "HH", "mm", "ss"):
            fmt = fmt.replace(key, token_map[key])
        return fmt
    prefix = re.sub(r"%date:([^%]+)%", repl, prefix)
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, output_dir
        )
        os.makedirs(full_output_folder, exist_ok=True)
        while True:
            basename = f"{filename}_{counter:05d}.{extension}"
            path = os.path.join(full_output_folder, basename)
            if not os.path.exists(path):
                break
            counter += 1
        display = os.path.join(subfolder, basename) if subfolder else basename
        return path, display.replace("\\", "/")
    except Exception:
        from pathlib import Path
        import time

        out_dir = Path.cwd() / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = prefix.replace("\\", "_").replace("/", "_").strip() or "AetherScale_MFG"
        basename = f"{safe}_{int(time.time())}.{extension}"
        return str(out_dir / basename), basename


def _frame_to_rgb24_bytes(frame_hwc: torch.Tensor) -> bytes:
    """Convert one normalized HWC frame directly to packed RGB24.

    Avoids VHS's float NumPy intermediate (`tensor.cpu().numpy() * 255`) and
    never materializes more than a single encoded frame on CPU.
    """
    x = frame_hwc[..., :3].detach()
    if not x.is_floating_point():
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)
    else:
        # Quantize before the GPU->CPU transfer when the frame is on CUDA.
        x = (x.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    if x.device.type != "cpu":
        x = x.to("cpu", non_blocking=False)
    return x.contiguous().numpy().tobytes()


def _parse_nvenc_gpu_choice(value: str) -> int | None:
    value = str(value or "auto").strip()
    if value == "auto":
        return None
    m = re.match(r"gpu_(\d+)", value)
    if m:
        return int(m.group(1))
    if value.isdigit():
        return int(value)
    return None


def _nvenc_env_for_choice(value: str) -> tuple[dict[str, str], int | None, dict[str, Any] | None]:
    env = os.environ.copy()
    idx = _parse_nvenc_gpu_choice(value)
    if idx is None:
        return env, None, None
    selected = None
    for gpu in _nvidia_smi_gpus():
        if int(gpu.get("index", -1)) == idx:
            selected = dict(gpu)
            break
    if selected is None:
        selected = {"index": idx, "name": f"GPU {idx}", "uuid": ""}
    selector = str(selected.get("uuid") or idx)
    env["CUDA_VISIBLE_DEVICES"] = selector
    return env, 0, selected


def _probe_nvenc_encoder(
    ffmpeg: str,
    encoder: str,
    env: dict[str, str],
    logical_gpu: int | None,
    *,
    width: int,
    height: int,
    pixel_format: str,
) -> tuple[bool, str]:
    probe_w = max(1, int(width))
    probe_h = max(1, int(height))
    if str(pixel_format) in {"yuv420p", "yuv422p", "yuv422p10le"}:
        probe_w += probe_w & 1
        probe_h += probe_h & 1
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={probe_w}x{probe_h}:r=1",
        "-frames:v", "1", "-c:v", str(encoder), "-pix_fmt", str(pixel_format),
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
    return cp.returncode == 0, cp.stderr.decode("utf-8", errors="replace")[-2500:]


def _native_video_encoder_args(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: float,
    output_path: str,
    codec: str,
    preset: str,
    bitrate_mbps: int,
    pixel_format: str,
    nvenc_gpu: str,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    encoder = str(codec)
    input_pix_fmt = "rgb24"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", input_pix_fmt,
        "-s:v", f"{int(width)}x{int(height)}", "-r", f"{float(fps):.8f}",
        "-i", "-", "-an",
    ]
    env = os.environ.copy()
    gpu_stats: dict[str, Any] = {"requested": str(nvenc_gpu), "logical_gpu": None, "selected": None}
    if encoder.endswith("_nvenc"):
        env, logical_gpu, selected = _nvenc_env_for_choice(str(nvenc_gpu))
        print(
            f"[AetherScale] MFG Video NVENC preflight: {width}x{height} {pixel_format} on {nvenc_gpu}",
            flush=True,
        )
        ok, detail = _probe_nvenc_encoder(
            ffmpeg, encoder, env, logical_gpu,
            width=width, height=height, pixel_format=pixel_format,
        )
        if not ok and logical_gpu is not None:
            # If UUID isolation is rejected by this FFmpeg build, retry its default adapter.
            env = os.environ.copy()
            logical_gpu = None
            ok, detail2 = _probe_nvenc_encoder(
                ffmpeg, encoder, env, None,
                width=width, height=height, pixel_format=pixel_format,
            )
            detail = detail + "\n--- auto fallback ---\n" + detail2
        if not ok:
            raise RuntimeError("NVENC preflight failed before native MFG video streaming:\n" + detail)
        cmd += ["-c:v", encoder]
        if logical_gpu is not None:
            cmd += ["-gpu", str(int(logical_gpu))]
        cmd += ["-preset", str(preset), "-rc", "vbr", "-b:v", f"{int(bitrate_mbps)}M"]
        gpu_stats = {
            "requested": str(nvenc_gpu),
            "logical_gpu": logical_gpu,
            "selected": selected,
            "preflight": "ok",
            "detail": detail,
        }
    elif encoder == "libx264":
        cpu_preset = {
            "p1": "ultrafast", "p2": "superfast", "p3": "veryfast", "p4": "faster",
            "p5": "fast", "p6": "medium", "p7": "slow",
        }.get(str(preset), "faster")
        cmd += ["-c:v", "libx264", "-preset", cpu_preset, "-b:v", f"{int(bitrate_mbps)}M"]
    elif encoder.startswith("prores_"):
        profiles = {
            "prores_proxy": 0, "prores_lt": 1, "prores_standard": 2,
            "prores_hq": 3, "prores_4444": 4, "prores_4444_xq": 5,
        }
        cmd += ["-c:v", "prores_ks", "-profile:v", str(profiles.get(encoder, 3))]
        pixel_format = "yuv422p10le" if encoder not in {"prores_4444", "prores_4444_xq"} else "yuv444p10le"
    else:
        cmd += ["-c:v", encoder]
    cmd += ["-pix_fmt", str(pixel_format), "-movflags", "+faststart", output_path]
    return cmd, env, gpu_stats


def _encode_native_dlssg_video(
    images: torch.Tensor,
    *,
    source_frame_rate: float,
    multiplier: int,
    guide_source: str,
    motion: Optional[MotionPacket],
    scene_cut_threshold: float,
    auto_bootstrap: bool,
    codec: str,
    preset: str,
    bitrate_mbps: int,
    filename_prefix: str,
    pixel_format: str,
    nvenc_gpu: str,
) -> dict[str, Any]:
    import time

    n, h, w, _c = [int(x) for x in images.shape]
    ffmpeg = _find_ffmpeg()
    output_fps = float(source_frame_rate) * int(multiplier)
    extension = "mov" if str(codec).startswith("prores_") else "mp4"
    output_path, display_name = _resolve_video_output_path(filename_prefix, extension)
    cmd, env, gpu_stats = _native_video_encoder_args(
        ffmpeg,
        width=w,
        height=h,
        fps=output_fps,
        output_path=output_path,
        codec=str(codec),
        preset=str(preset),
        bitrate_mbps=int(bitrate_mbps),
        pixel_format=str(pixel_format),
        nvenc_gpu=str(nvenc_gpu),
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=creationflags,
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin pipe for native MFG video.")
    encoded = 0
    start = time.perf_counter()
    expected = (n - 1) * int(multiplier) + 1
    console_progress = ConsoleProgress("MFG Video / native DLSSG + encode", expected, unit="frame")

    def emit(rgba, index, ideal, frame):
        nonlocal encoded
        del index, ideal, frame
        rgb = np.ascontiguousarray(rgba[..., :3], dtype=np.uint8)
        proc.stdin.write(memoryview(rgb).cast("B"))
        encoded += 1
        console_progress.update(1)

    try:
        native_stats = stream_native_mfg(
            images,
            source_frame_rate=float(source_frame_rate),
            multiplier=int(multiplier),
            emit=emit,
            guide_source=str(guide_source),
            motion=motion,
            scene_cut_threshold=float(scene_cut_threshold),
            auto_bootstrap=bool(auto_bootstrap),
        )
        proc.stdin.flush()
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"FFmpeg failed after native DLSSG generation (exit {returncode}):\n{detail}")
    except Exception:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    console_progress.close(status="done")
    elapsed = max(1e-9, time.perf_counter() - start)
    size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    native_stats.update({
        "output_path": output_path,
        "display_name": display_name,
        "codec": str(codec),
        "preset": str(preset),
        "bitrate_mbps": int(bitrate_mbps),
        "pixel_format": str(pixel_format),
        "streaming_direct_to_ffmpeg": True,
        "full_generated_image_batch_created": False,
        "encoded_frames": int(encoded),
        "elapsed_seconds": round(elapsed, 3),
        "encode_pipeline_fps": round(encoded / elapsed, 3),
        "output_bytes": int(size_bytes),
        "ffmpeg": ffmpeg,
        "nvenc": gpu_stats,
    })
    return native_stats


def _resolve_motion_packet_for_mfg(
    images: torch.Tensor,
    motion: Optional[MotionPacket],
    motion_source: str,
    cuda_device: int,
) -> tuple[Optional[MotionPacket], str]:
    packet = motion
    resolved = str(motion_source)
    if packet is not None and packet.flow.numel() > 0:
        return packet, "connected_motion"
    if motion_source == "connected_motion":
        return None, "none_available"
    if motion_source == "rgb_only":
        return None, "rgb_only"
    if torch.cuda.is_available():
        packet = analyze_motion(
            images,
            cuda_device=int(cuda_device),
            engine="torch_lk",
            quality="mfg_safe",
            scene_cut_threshold=0.24,
            reset_on_scene_cut=True,
            output_device="cpu_safe",
            motion_mode="compact_flow",
            analysis_long_edge=512,
            storage_precision="float16",
        )
        return packet, "internal_compact"
    return None, "rgb_only_fallback_no_cuda"


def encode_mfg_video(
    images: torch.Tensor,
    *,
    motion: Optional[MotionPacket],
    source_frame_rate: float,
    multiplier: int,
    motion_source: str,
    surrogate_method: str,
    scene_cut_strategy: str,
    cuda_device: int,
    codec: str,
    preset: str,
    bitrate_mbps: int,
    filename_prefix: str,
    pixel_format: str = "yuv420p",
    artifact_guard: float = 1.0,
    emissive_protection: float = 0.85,
    thin_detail_protection: float = 0.75,
    mv_confidence_threshold: float = 0.45,
    fallback_mode: str = "closest_source",
    synthesis_mode: str = "continuous_temporal",
    backend: str = "native_dlssg",
    native_auto_bootstrap: bool = True,
    native_guide_source: str = "internal_dis",
    native_scene_cut_threshold: float = 0.24,
    nvenc_gpu: str = "auto",
) -> dict[str, Any]:
    """Generate MFG-Lab frames and stream them directly into FFmpeg.

    The generated sequence never exists as a full ComfyUI IMAGE tensor. This is
    the fast path for long 2x/3x/4x jobs: generate -> RGB24 -> FFmpeg/NVENC.
    """
    import time

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(f"Expected IMAGE [T,H,W,C], got {tuple(images.shape)}")
    n, h, w, c = [int(x) for x in images.shape]
    if n < 1:
        raise ValueError("MFG Video received an empty IMAGE batch.")
    if int(multiplier) < 1:
        raise ValueError("multiplier must be >= 1")
    if float(source_frame_rate) <= 0:
        raise ValueError("source_frame_rate must be > 0")

    if str(backend) == "native_dlssg":
        return _encode_native_dlssg_video(
            images,
            source_frame_rate=float(source_frame_rate),
            multiplier=int(multiplier),
            guide_source=str(native_guide_source),
            motion=motion,
            scene_cut_threshold=float(native_scene_cut_threshold),
            auto_bootstrap=bool(native_auto_bootstrap),
            codec=str(codec),
            preset=str(preset),
            bitrate_mbps=int(bitrate_mbps),
            filename_prefix=str(filename_prefix),
            pixel_format=str(pixel_format),
            nvenc_gpu=str(nvenc_gpu),
        )

    ffmpeg = _find_ffmpeg()
    output_fps = float(source_frame_rate) * float(multiplier)
    output_frames = (n - 1) * int(multiplier) + 1
    extension = "mp4"
    output_path, display_name = _resolve_video_output_path(filename_prefix, extension)

    encoder = str(codec)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s:v", f"{w}x{h}",
        "-r", f"{output_fps:.8f}",
        "-i", "-",
        "-an",
        "-c:v", encoder,
    ]
    if encoder.endswith("_nvenc"):
        cmd += ["-preset", str(preset), "-rc", "vbr", "-b:v", f"{int(bitrate_mbps)}M"]
    elif encoder == "libx264":
        cpu_preset = {
            "p1": "ultrafast", "p2": "superfast", "p3": "veryfast", "p4": "faster",
            "p5": "fast", "p6": "medium", "p7": "slow",
        }.get(str(preset), "faster")
        cmd += ["-preset", cpu_preset, "-b:v", f"{int(bitrate_mbps)}M"]
    cmd += ["-pix_fmt", str(pixel_format), "-movflags", "+faststart", output_path]

    packet, resolved_motion_source = _resolve_motion_packet_for_mfg(
        images, motion, str(motion_source), int(cuda_device)
    )
    work_device = (
        torch.device(f"cuda:{int(cuda_device)}") if torch.cuda.is_available() else torch.device("cpu")
    )

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    start = time.perf_counter()
    encoded = 0
    inserted = 0
    cut_pairs = 0
    guarded_pairs = 0
    unstable_ratio_sum = 0.0
    emissive_ratio_sum = 0.0
    detail_ratio_sum = 0.0
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin pipe.")

    console_progress = ConsoleProgress("MFG Video / surrogate + encode", output_frames, unit="frame")

    def push(frame_hwc: torch.Tensor) -> None:
        nonlocal encoded
        proc.stdin.write(_frame_to_rgb24_bytes(frame_hwc))
        encoded += 1
        console_progress.update(1)

    try:
        if n == 1:
            push(images[0])
        else:
            for i in range(n - 1):
                prev_hwc = images[i]
                curr_hwc = images[i + 1]
                push(prev_hwc)

                cut = bool(packet.scene_cuts[i].item()) if packet is not None and packet.scene_cuts.numel() else False
                if cut:
                    cut_pairs += 1

                prev = prev_hwc.to(device=work_device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()
                curr = curr_hwc.to(device=work_device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()

                flow = conf = guard_context = None
                if not cut and packet is not None and packet.flow.numel() > 0:
                    flow = _upsample_flow(packet.flow[i].to("cpu"), h, w, work_device)
                    if packet.confidence.numel():
                        conf = _upsample_conf(packet.confidence[i].to("cpu"), h, w, work_device)
                    else:
                        conf = torch.ones((1, 1, h, w), device=work_device, dtype=torch.float32)
                    guard_context = _build_guard_context(
                        prev,
                        curr,
                        flow,
                        conf,
                        artifact_guard=float(artifact_guard),
                        emissive_protection=float(emissive_protection),
                        thin_detail_protection=float(thin_detail_protection),
                        mv_confidence_threshold=float(mv_confidence_threshold),
                    )
                    guarded_pairs += 1
                    unstable_ratio_sum += float(guard_context['unstable'].mean().item())
                    emissive_ratio_sum += float(guard_context['emissive_risk'].mean().item())
                    detail_ratio_sum += float(guard_context['detail_risk'].mean().item())

                for sub in range(1, int(multiplier)):
                    t = sub / float(multiplier)
                    if cut:
                        frame_nchw = _scene_cut_fill(prev, curr, t, scene_cut_strategy)
                    elif surrogate_method == "frame_repeat":
                        frame_nchw = prev
                    elif surrogate_method == "linear_blend" or flow is None or conf is None:
                        frame_nchw = _linear_tween(prev, curr, t)
                    else:
                        frame_nchw = _mv_surrogate(
                            prev, curr, flow, conf, t,
                            guard_context=guard_context,
                            fallback_mode=str(fallback_mode),
                            synthesis_mode=str(synthesis_mode),
                        )
                    push(frame_nchw[0].permute(1, 2, 0))
                    inserted += 1

                del prev, curr
                if flow is not None:
                    del flow
                if conf is not None:
                    del conf
                if guard_context is not None:
                    del guard_context
            push(images[-1])

        proc.stdin.flush()
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[-4000:]
            try:
                os.remove(output_path)
            except OSError:
                pass
            raise RuntimeError(f"FFmpeg/NVENC failed with exit code {returncode}:\n{detail}")
    except BrokenPipeError as exc:
        detail = b""
        try:
            detail = proc.stderr.read() if proc.stderr is not None else b""
        except Exception:
            pass
        proc.kill()
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise RuntimeError(
            "FFmpeg closed the input pipe early: " + detail.decode("utf-8", errors="replace")[-4000:]
        ) from exc
    except Exception:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        raise

    console_progress.close(status="done")
    elapsed = max(1e-9, time.perf_counter() - start)
    size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    return {
        "output_path": output_path,
        "display_name": display_name,
        "codec": encoder,
        "preset": str(preset),
        "bitrate_mbps": int(bitrate_mbps),
        "pixel_format": str(pixel_format),
        "source_frame_rate": float(source_frame_rate),
        "output_frame_rate": float(output_fps),
        "multiplier": int(multiplier),
        "input_frames": n,
        "output_frames": int(encoded),
        "expected_output_frames": int(output_frames),
        "inserted_frames": int(inserted),
        "scene_cut_pairs": int(cut_pairs),
        "backend": "surrogate_mv",
        "motion_source_resolved": resolved_motion_source,
        "surrogate_method": str(surrogate_method),
        "artifact_guard": float(artifact_guard),
        "emissive_protection": float(emissive_protection),
        "thin_detail_protection": float(thin_detail_protection),
        "mv_confidence_threshold": float(mv_confidence_threshold),
        "fallback_mode": str(fallback_mode),
        "synthesis_mode": str(synthesis_mode),
        "continuous_temporal_branch_policy": "source_locked_prev_no_midpoint_switch",
        "temporal_positions": [round(i / float(multiplier), 6) for i in range(1, int(multiplier))],
        "guarded_pairs": int(guarded_pairs),
        "avg_unstable_ratio": round(unstable_ratio_sum / max(guarded_pairs, 1), 6),
        "avg_emissive_risk": round(emissive_ratio_sum / max(guarded_pairs, 1), 6),
        "avg_detail_risk": round(detail_ratio_sum / max(guarded_pairs, 1), 6),
        "streaming_direct_to_ffmpeg": True,
        "full_generated_image_batch_created": False,
        "elapsed_seconds": round(elapsed, 3),
        "encode_pipeline_fps": round(encoded / elapsed, 3),
        "output_bytes": int(size_bytes),
        "ffmpeg": ffmpeg,
    }
