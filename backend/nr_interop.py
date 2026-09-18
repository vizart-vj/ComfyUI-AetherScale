from __future__ import annotations

"""AetherScale in-process DLSS Neural Rendering adapter.

This module is AetherScale-owned integration code. It intentionally treats the
MIT-licensed OreX native bridge as a third-party binary dependency behind a
small ABI adapter instead of copying its Python implementation. The adapter
adds AetherScale-specific GPU matching, bounded chunking, mmap/spill storage,
interrupts, progress reporting, and explicit fallback behavior.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import threading
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .download import download_file
from .progress import ConsoleProgress, throw_if_interrupted
from .storage import StorageInfo, allocate_cpu_tensor, resolve_dtype, sync_file_backed_tensor
from . import dlssnr as legacy_nr


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "runtime" / "nr_interop"
VENDOR_BRIDGE_DIR = VENDOR_ROOT / "bridge"
VENDOR_RUNTIME_DIR = VENDOR_ROOT / "runtime"
VENDOR_CALLER_DIR = VENDOR_RUNTIME_DIR / "caller"
VENDOR_BRIDGE = VENDOR_BRIDGE_DIR / "dlss5nr_bridge.dll"
VENDOR_CALLER = VENDOR_CALLER_DIR / "nvngx.dll_comfy.dll"
VENDOR_NR_DLL = VENDOR_RUNTIME_DIR / "nvngx_dlssnr.dll"
VENDOR_CORE_DLL = VENDOR_RUNTIME_DIR / "_nvngx.dll"
VENDOR_MANIFEST = VENDOR_ROOT / "manifest.json"

# We pin to one public commit and verify the Git blob identity after download.
# That is stronger than trusting mutable `main` content while avoiding shipping
# third-party binaries in the AetherScale archive.
OREX_COMMIT = "739208e7ae5f576355fc5c30ffb77c4de2e61984"
OREX_BRIDGE_URL = (
    f"https://raw.githubusercontent.com/orex2121/ComfyUI-DLSS5-orex/"
    f"{OREX_COMMIT}/bridge/bin/dlss5nr_bridge.dll"
)
OREX_BRIDGE_ALT_URL = (
    f"https://github.com/orex2121/ComfyUI-DLSS5-orex/raw/"
    f"{OREX_COMMIT}/bridge/bin/dlss5nr_bridge.dll"
)
OREX_CALLER_URL = (
    f"https://raw.githubusercontent.com/orex2121/ComfyUI-DLSS5-orex/"
    f"{OREX_COMMIT}/runtime/caller/nvngx.dll_comfy.dll"
)
OREX_CALLER_ALT_URL = (
    f"https://github.com/orex2121/ComfyUI-DLSS5-orex/raw/"
    f"{OREX_COMMIT}/runtime/caller/nvngx.dll_comfy.dll"
)
OREX_BRIDGE_BLOB_SHA1 = "ffd67747b1272607753743907369e4fb570b0efe"
OREX_CALLER_BLOB_SHA1 = "c69856b68a67a795de27c307919adf2ec7dd0ac2"
OREX_BRIDGE_SIZE = 238_592
OREX_CALLER_SIZE = 103_424
VERIFIED_NR_URL = "https://github.com/orex2121/ComfyUI-DLSS5-orex/releases/download/nvngx-v1/nvngx_dlssnr.dll"
VERIFIED_NR_API_URL = "https://api.github.com/repos/orex2121/ComfyUI-DLSS5-orex/releases/assets/546953587"
VERIFIED_NR_SHA256 = "6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927"
VERIFIED_NR_SIZE = 165_830_144

_UPSCALE_FACTORS = {
    "native_1x": 1.0,
    "quality_1_5x": 1.5,
    "balanced_1_724x": 1.724,
    "performance_2x": 2.0,
    "ultra_performance_3x": 3.0,
}
_MAX_LONG_EDGE = 7680
_MAX_SHORT_EDGE = 4320

_lock = threading.RLock()
_lib: Any | None = None
_initialized_bridge_gpu: int | None = None
_initialized_runtime_fingerprint: str | None = None
_dll_dir_handles: list[Any] = []
_gpu_mapping_cache: dict[str, int] = {}
_channel_swap_cache: dict[tuple[int, int, int], bool] = {}


class NativeNRInteropError(RuntimeError):
    pass


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _verify_blob(path: Path, *, expected_size: int, expected_sha1: str, label: str) -> None:
    if not path.is_file():
        raise NativeNRInteropError(f"{label} is missing: {path}")
    size = path.stat().st_size
    if size != int(expected_size):
        raise NativeNRInteropError(
            f"{label} size mismatch: expected {expected_size}, got {size}."
        )
    digest = _git_blob_sha1(path)
    if digest != expected_sha1.lower():
        raise NativeNRInteropError(
            f"{label} Git blob checksum mismatch: expected {expected_sha1}, got {digest}."
        )


def _download_vendor_file(
    path: Path, url: str, expected_size: int, expected_sha1: str, label: str, *, extra_urls: tuple[str, ...] = ()
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            _verify_blob(path, expected_size=expected_size, expected_sha1=expected_sha1, label=label)
            return {"transport": "existing", "url": url}
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
    try:
        result = download_file(
            url,
            path,
            user_agent="ComfyUI-AetherScale/0.9.2",
            timeout=180,
            extra_urls=extra_urls,
        )
        _verify_blob(path, expected_size=expected_size, expected_sha1=expected_sha1, label=label)
        return result
    except NativeNRInteropError:
        raise
    except Exception as exc:
        raise NativeNRInteropError(f"Failed to bootstrap {label}: {type(exc).__name__}: {exc}") from exc


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.stat().st_size == dst.stat().st_size:
                return "existing"
        except OSError:
            pass
        try:
            dst.unlink()
        except OSError:
            pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            shutil.copy2(src, dst)
            return "copy"
        except OSError as exc:
            raise NativeNRInteropError(
                f"Could not stage native NR runtime file {src} -> {dst}: {exc}"
            ) from exc


def _resolve_runtime_source(custom_path: str, auto_bootstrap: bool, gpu_index: int) -> tuple[Path, dict[str, Any]]:
    custom = Path(custom_path).expanduser() if str(custom_path or "").strip() else None
    if custom is not None:
        if custom.is_dir():
            custom = custom / "nvngx_dlssnr.dll"
        if not custom.is_file():
            raise NativeNRInteropError(f"Custom DLSSNR runtime was not found: {custom}")
        return custom.resolve(), {"source": "custom", "sha256": _sha256_file(custom)}

    # Prefer the runtime build verified by the in-process bridge project instead
    # of reusing AetherScale's older stock feature-18 diagnostic runtime. This
    # removes a major source of BAD00001 incompatibility on RTX 50 systems.
    if VENDOR_NR_DLL.is_file():
        try:
            digest = _sha256_file(VENDOR_NR_DLL)
            if VENDOR_NR_DLL.stat().st_size == VERIFIED_NR_SIZE and digest == VERIFIED_NR_SHA256:
                return VENDOR_NR_DLL.resolve(), {
                    "source": "existing_verified_native_runtime",
                    "sha256": digest,
                }
        except OSError:
            pass

    # Reuse an already-installed matching DLL without duplicating 158 MiB.
    for candidate in (legacy_nr.NR_DLL,):
        if candidate.is_file():
            try:
                digest = _sha256_file(candidate)
                if candidate.stat().st_size == VERIFIED_NR_SIZE and digest == VERIFIED_NR_SHA256:
                    return candidate.resolve(), {
                        "source": "existing_verified_legacy_runtime",
                        "sha256": digest,
                    }
            except OSError:
                pass

    if auto_bootstrap:
        VENDOR_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        print("[AetherScale] Native NR: downloading pinned verified 310.8.SF-v2 runtime...", flush=True)
        try:
            download_file(
                VERIFIED_NR_URL,
                VENDOR_NR_DLL,
                user_agent="ComfyUI-AetherScale/0.9.2",
                timeout=300,
                extra_urls=(VERIFIED_NR_API_URL,),
                headers={"Accept": "application/octet-stream"},
            )
        except Exception as exc:
            raise NativeNRInteropError(
                f"Failed to bootstrap the pinned native NR runtime: {type(exc).__name__}: {exc}"
            ) from exc
        size = VENDOR_NR_DLL.stat().st_size
        digest = _sha256_file(VENDOR_NR_DLL)
        if size != VERIFIED_NR_SIZE or digest != VERIFIED_NR_SHA256:
            try:
                VENDOR_NR_DLL.unlink()
            except OSError:
                pass
            raise NativeNRInteropError(
                "Pinned native NR runtime checksum mismatch: "
                f"expected {VERIFIED_NR_SHA256}/{VERIFIED_NR_SIZE} bytes, got {digest}/{size}."
            )
        return VENDOR_NR_DLL.resolve(), {
            "source": "downloaded_verified_orex_release",
            "url": VERIFIED_NR_URL,
            "sha256": digest,
        }

    # With bootstrap disabled, allow an existing AetherScale-discovered runtime
    # as an explicit compatibility choice, but surface that it is unverified for
    # this bridge rather than pretending all feature-18 DLLs are equivalent.
    discovered = legacy_nr.discover_runtime(custom_path="", stage=True)
    if discovered is None:
        raise NativeNRInteropError(
            "No native NR runtime is available. Enable auto_bootstrap or provide runtime_path."
        )
    path = Path(discovered).resolve()
    return path, {
        "source": "existing_unverified_compat_runtime",
        "sha256": _sha256_file(path),
        "warning": "This runtime is not the pinned 310.8.SF-v2 build verified for native_interop.",
    }


def ensure_native_bundle(*, custom_path: str = "", auto_bootstrap: bool = True, gpu_index: int = 0) -> dict[str, Any]:
    if platform.system() != "Windows":
        raise NativeNRInteropError("Native DLSS Neural Rendering interop requires Windows/D3D12.")

    bridge_dl = _download_vendor_file(
        VENDOR_BRIDGE,
        OREX_BRIDGE_URL,
        OREX_BRIDGE_SIZE,
        OREX_BRIDGE_BLOB_SHA1,
        "OreX MIT bridge",
        extra_urls=(OREX_BRIDGE_ALT_URL,),
    )
    caller_dl = _download_vendor_file(
        VENDOR_CALLER,
        OREX_CALLER_URL,
        OREX_CALLER_SIZE,
        OREX_CALLER_BLOB_SHA1,
        "OreX MIT caller shim",
        extra_urls=(OREX_CALLER_ALT_URL,),
    )

    runtime_src, runtime_meta = _resolve_runtime_source(custom_path, auto_bootstrap, int(gpu_index))
    runtime_link = _link_or_copy(runtime_src, VENDOR_NR_DLL) if runtime_src != VENDOR_NR_DLL.resolve() else "existing"

    # The bridge can normally find the NGX core through DriverStore. A local
    # override is still useful on machines that retain multiple driver packages.
    core_stage = legacy_nr._stage_best_ngx_core(int(gpu_index))
    core_link = None
    if legacy_nr.CORE_DLL.is_file():
        core_link = _link_or_copy(legacy_nr.CORE_DLL, VENDOR_CORE_DLL)

    manifest = {
        "backend": "AetherScale native NR interop adapter",
        "third_party_bridge": "orex2121/ComfyUI-DLSS5-orex",
        "third_party_commit": OREX_COMMIT,
        "bridge_git_blob_sha1": OREX_BRIDGE_BLOB_SHA1,
        "caller_git_blob_sha1": OREX_CALLER_BLOB_SHA1,
        "bridge_transport": bridge_dl,
        "caller_transport": caller_dl,
        "runtime_source": str(runtime_src),
        "runtime_meta": runtime_meta,
        "runtime_link_mode": runtime_link,
        "core_link_mode": core_link,
        "core_stage": core_stage,
    }
    try:
        VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
        VENDOR_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        pass
    return manifest


def _register_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for p in (VENDOR_BRIDGE_DIR, VENDOR_RUNTIME_DIR, VENDOR_CALLER_DIR):
        if p.is_dir():
            try:
                _dll_dir_handles.append(os.add_dll_directory(str(p)))
            except OSError:
                pass


def _load_library() -> Any:
    global _lib
    if _lib is not None:
        return _lib
    if platform.system() != "Windows":
        raise NativeNRInteropError("Native DLSS Neural Rendering interop requires Windows/D3D12.")
    if not VENDOR_BRIDGE.is_file():
        raise NativeNRInteropError(f"Native NR bridge is missing: {VENDOR_BRIDGE}")

    _register_dll_dirs()
    lib = ctypes.WinDLL(str(VENDOR_BRIDGE))
    lib.dlss5nr_init.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int]
    lib.dlss5nr_init.restype = ctypes.c_int
    lib.dlss5nr_process.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.dlss5nr_process.restype = ctypes.c_int
    lib.dlss5nr_process_cuda.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.dlss5nr_process_cuda.restype = ctypes.c_int
    lib.dlss5nr_cuda_supported.argtypes = []
    lib.dlss5nr_cuda_supported.restype = ctypes.c_int
    lib.dlss5nr_cuda_status.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.dlss5nr_cuda_status.restype = ctypes.c_int
    lib.dlss5nr_shutdown.argtypes = []
    lib.dlss5nr_shutdown.restype = None
    lib.dlss5nr_version.argtypes = []
    lib.dlss5nr_version.restype = ctypes.c_char_p
    lib.dlss5nr_gpu_name.argtypes = []
    lib.dlss5nr_gpu_name.restype = ctypes.c_char_p
    _lib = lib
    return lib


def _decode_error(buf: ctypes.Array) -> str:
    try:
        return buf.value.decode("utf-8", errors="replace")
    except Exception:
        return "Unknown native NR bridge error"


def _bridge_gpu_name(lib: Any) -> str:
    try:
        raw = lib.dlss5nr_gpu_name()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"
    except Exception:
        return "unknown"


def _normalized_gpu_name(name: str) -> str:
    text = str(name or "").lower()
    for token in ("nvidia", "geforce", "graphics", "adapter"):
        text = text.replace(token, " ")
    return " ".join(text.split())


def _gpu_names_match(a: str, b: str) -> bool:
    na, nb = _normalized_gpu_name(a), _normalized_gpu_name(b)
    return bool(na and nb and (na == nb or na in nb or nb in na))


def _shutdown_unlocked(lib: Any) -> None:
    global _initialized_bridge_gpu, _initialized_runtime_fingerprint
    try:
        lib.dlss5nr_shutdown()
    except Exception:
        pass
    _initialized_bridge_gpu = None
    _initialized_runtime_fingerprint = None


def shutdown() -> None:
    with _lock:
        if _lib is not None:
            _shutdown_unlocked(_lib)


def _runtime_fingerprint() -> str:
    parts = []
    for p in (VENDOR_BRIDGE, VENDOR_CALLER, VENDOR_NR_DLL, VENDOR_CORE_DLL):
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{p.name}:missing")
    return "|".join(parts)


def _init_candidate(lib: Any, candidate: int) -> tuple[bool, str, str]:
    err = ctypes.create_string_buffer(4096)
    ok = bool(lib.dlss5nr_init(int(candidate), str(VENDOR_RUNTIME_DIR), err, len(err)))
    if not ok:
        return False, "unknown", _decode_error(err)
    return True, _bridge_gpu_name(lib), ""


def _resolve_bridge_gpu(lib: Any, torch_gpu_index: int) -> tuple[int, str, list[str]]:
    target = ""
    if torch.cuda.is_available() and 0 <= int(torch_gpu_index) < torch.cuda.device_count():
        target = torch.cuda.get_device_name(int(torch_gpu_index))
    key = _normalized_gpu_name(target) or f"cuda:{torch_gpu_index}"
    cached = _gpu_mapping_cache.get(key)
    candidates: list[int] = []
    if cached is not None:
        candidates.append(int(cached))
    # Start with the CUDA ordinal, then scan a small DXGI ordinal range. This is
    # important on dual-GPU systems where DXGI order and CUDA_VISIBLE_DEVICES differ.
    for candidate in [int(torch_gpu_index), 0, 1, 2, 3, 4, 5, 6, 7]:
        if candidate >= 0 and candidate not in candidates:
            candidates.append(candidate)

    attempts: list[str] = []
    for candidate in candidates:
        throw_if_interrupted()
        ok, name, error = _init_candidate(lib, candidate)
        if not ok:
            attempts.append(f"gpu {candidate}: {error}")
            _shutdown_unlocked(lib)
            continue
        if not target or _gpu_names_match(name, target):
            _gpu_mapping_cache[key] = candidate
            return candidate, name, attempts
        attempts.append(f"gpu {candidate}: initialized {name}, wanted {target}")
        _shutdown_unlocked(lib)
    raise NativeNRInteropError(
        "Could not map the requested PyTorch CUDA device to the native D3D12 bridge. "
        + " | ".join(attempts[-8:])
    )


def _ensure_session(torch_gpu_index: int, custom_path: str, auto_bootstrap: bool) -> tuple[Any, int, str, dict[str, Any]]:
    global _initialized_bridge_gpu, _initialized_runtime_fingerprint
    manifest = ensure_native_bundle(
        custom_path=custom_path,
        auto_bootstrap=bool(auto_bootstrap),
        gpu_index=int(torch_gpu_index),
    )
    lib = _load_library()
    fingerprint = _runtime_fingerprint()

    with _lock:
        # Re-probe only when runtime files changed or no compatible session exists.
        if _initialized_bridge_gpu is not None and _initialized_runtime_fingerprint == fingerprint:
            current_name = _bridge_gpu_name(lib)
            target_name = (
                torch.cuda.get_device_name(int(torch_gpu_index))
                if torch.cuda.is_available() and int(torch_gpu_index) < torch.cuda.device_count()
                else ""
            )
            if not target_name or _gpu_names_match(current_name, target_name):
                return lib, int(_initialized_bridge_gpu), current_name, manifest
            _shutdown_unlocked(lib)

        bridge_gpu, bridge_name, attempts = _resolve_bridge_gpu(lib, int(torch_gpu_index))
        _initialized_bridge_gpu = int(bridge_gpu)
        _initialized_runtime_fingerprint = fingerprint
        manifest["gpu_probe_attempts"] = attempts
        return lib, bridge_gpu, bridge_name, manifest


def _style_to_int(style: str) -> int:
    return {
        "auto": 1,
        "natural": 1,
        "cinematic": 2,
        "material_detail": 0,
        "default": 0,
    }.get(str(style), int(style) if str(style).isdigit() else 0)


def _target_size(width: int, height: int, upscale_mode: str) -> tuple[int, int, float]:
    factor = float(_UPSCALE_FACTORS.get(str(upscale_mode), 1.0))
    if factor == 1.0:
        return int(width), int(height), factor
    out_w = max(2, int(round(width * factor)))
    out_h = max(2, int(round(height * factor)))
    out_w += out_w & 1
    out_h += out_h & 1
    if max(out_w, out_h) > _MAX_LONG_EDGE or min(out_w, out_h) > _MAX_SHORT_EDGE:
        raise NativeNRInteropError(
            f"Native NR target {out_w}x{out_h} exceeds the supported 7680x4320 boundary."
        )
    return out_w, out_h, factor


def _scene_cut_flags(images: torch.Tensor, motion: Optional[Any], enabled: bool, threshold: float) -> list[bool]:
    count = int(images.shape[0])
    if count <= 1 or not enabled:
        return [False] * max(0, count - 1)
    if motion is not None:
        try:
            values = [bool(v) for v in motion.scene_cuts.detach().cpu().tolist()]
            if len(values) >= count - 1:
                return values[: count - 1]
        except Exception:
            pass

    flags: list[bool] = []
    prev = None
    # Work on a tiny 64x64 luma proxy. This is deliberately AetherScale-specific
    # and independent of any third-party scene detector implementation.
    for i in range(count):
        throw_if_interrupted()
        frame = images[i, ..., :3].detach().to(dtype=torch.float32)
        if frame.device.type != "cpu":
            t = frame.permute(2, 0, 1).unsqueeze(0)
            small = F.interpolate(t, size=(64, 64), mode="area")[0]
            gray = (0.299 * small[0] + 0.587 * small[1] + 0.114 * small[2]).mean().item()
            # A scalar mean alone misses spatial cuts; add block mean grid.
            grid = F.adaptive_avg_pool2d(t, (8, 8))[0].mean(dim=0).detach().cpu()
        else:
            t = frame.permute(2, 0, 1).unsqueeze(0)
            small = F.interpolate(t, size=(64, 64), mode="area")[0]
            gray = float((0.299 * small[0] + 0.587 * small[1] + 0.114 * small[2]).mean())
            grid = F.adaptive_avg_pool2d(t, (8, 8))[0].mean(dim=0)
        current = (gray, grid)
        if prev is not None:
            mean_delta = abs(current[0] - prev[0])
            spatial = float((current[1] - prev[1]).abs().mean())
            score = 0.35 * mean_delta + 0.65 * spatial
            flags.append(score >= float(threshold))
        prev = current
    return flags


def _cuda_status(lib: Any) -> str:
    buf = ctypes.create_string_buffer(1024)
    try:
        ok = bool(lib.dlss5nr_cuda_status(buf, len(buf)))
        text = buf.value.decode("utf-8", errors="replace")
        return ("ok: " if ok else "unavailable: ") + (text or "unknown")
    except Exception as exc:
        return f"status_error: {type(exc).__name__}: {exc}"


def _detect_channel_swap(reference: torch.Tensor, raw: torch.Tensor) -> bool:
    h, w = int(reference.shape[0]), int(reference.shape[1])
    sy = max(1, h // 128)
    sx = max(1, w // 128)
    ref = reference[::sy, ::sx]
    rr = raw[::sy, ::sx]
    sw = rr[..., [2, 1, 0]]
    raw_score = float((rr - ref).abs().mean()) + float((rr.mean(dim=(0, 1)) - ref.mean(dim=(0, 1))).abs().mean())
    sw_score = float((sw - ref).abs().mean()) + float((sw.mean(dim=(0, 1)) - ref.mean(dim=(0, 1))).abs().mean())
    return sw_score < raw_score


def _process_cuda_frame(
    lib: Any,
    frame_in: torch.Tensor,
    frame_out: torch.Tensor,
    *,
    style_i: int,
    preset: int,
    intensity: float,
    tone: float,
    structure: float,
    skin: float,
    auto_mask: bool,
    reset: int,
) -> None:
    err = ctypes.create_string_buffer(4096)
    stream_handle = int(torch.cuda.current_stream(frame_in.device).cuda_stream)
    ok = bool(lib.dlss5nr_process_cuda(
        ctypes.c_uint64(int(frame_in.data_ptr())),
        ctypes.c_uint64(int(frame_out.data_ptr())),
        int(frame_in.shape[1]),
        int(frame_in.shape[0]),
        ctypes.c_uint64(stream_handle),
        int(style_i),
        int(preset),
        ctypes.c_float(float(intensity)),
        ctypes.c_float(float(tone)),
        ctypes.c_float(float(structure)),
        ctypes.c_float(float(skin)),
        1 if auto_mask else 0,
        int(reset),
        err,
        len(err),
    ))
    if not ok:
        raise NativeNRInteropError(_decode_error(err))


def _process_cpu_frame(
    lib: Any,
    frame_in: torch.Tensor,
    *,
    style_i: int,
    preset: int,
    intensity: float,
    tone: float,
    structure: float,
    skin: float,
    auto_mask: bool,
    reset: int,
) -> torch.Tensor:
    cpu = frame_in.detach().to(device="cpu", dtype=torch.float32).contiguous()
    src = cpu.numpy()
    dst = np.empty_like(src, dtype=np.float32)
    err = ctypes.create_string_buffer(4096)
    ok = bool(lib.dlss5nr_process(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(src.shape[1]), int(src.shape[0]),
        int(style_i), int(preset),
        ctypes.c_float(float(intensity)), ctypes.c_float(float(tone)),
        ctypes.c_float(float(structure)), ctypes.c_float(float(skin)),
        1 if auto_mask else 0, int(reset), err, len(err),
    ))
    if not ok:
        raise NativeNRInteropError(_decode_error(err))
    return torch.from_numpy(dst)


def _choose_chunk_frames(
    requested: int,
    *,
    cuda_device: torch.device,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    safety_margin_mb: int,
    remaining: int,
) -> int:
    requested = max(1, int(requested))
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        return min(requested, remaining)
    try:
        free, _total = torch.cuda.mem_get_info(cuda_device)
    except Exception:
        return min(requested, remaining)
    margin = max(int(safety_margin_mb) * 1024 * 1024, 768 * 1024 * 1024)
    usable = max(0, int(free) - margin)
    # input float32 + output float32 + resize/interop headroom
    per_frame = (in_w * in_h * 3 * 4) + (out_w * out_h * 3 * 4 * 2)
    by_vram = max(1, usable // max(per_frame, 1))
    return max(1, min(requested, remaining, int(by_vram)))


def _storage_dict(storage: StorageInfo | None, *, output_bytes: int, gpu_direct: bool) -> dict[str, Any]:
    if gpu_direct:
        return {
            "output_storage_backend": "gpu_direct",
            "output_storage_path": None,
            "output_storage_fallback_reason": None,
            "output_gib": round(output_bytes / 1024**3, 3),
        }
    assert storage is not None
    return {
        "output_storage_backend": storage.backend,
        "output_storage_path": storage.path,
        "output_storage_fallback_reason": storage.fallback_reason,
        "output_gib": round(storage.bytes / 1024**3, 3),
        "system_commit_available_gib_at_allocation": (
            round(storage.commit_available_bytes / 1024**3, 3)
            if storage.commit_available_bytes is not None else None
        ),
        "spill_disk_free_gib_at_allocation": (
            round(storage.disk_free_bytes / 1024**3, 3)
            if storage.disk_free_bytes is not None else None
        ),
    }


def process_native_interop(
    images: torch.Tensor,
    *,
    motion: Optional[Any],
    style: str,
    preset: int,
    intensity: float,
    tone: float,
    structure: float,
    skin: float,
    auto_mask: bool,
    temporal_mode: str,
    reset_on_scene_cut: bool,
    scene_change_threshold: float,
    gpu_index: int,
    safety_margin_mb: int,
    vram_guard: str,
    output_device: str,
    auto_bootstrap: bool,
    runtime_path: str,
    output_precision: str,
    output_storage: str,
    clean_cache: bool,
    upscale_mode: str,
    chunk_frames: int,
    channel_order: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise NativeNRInteropError("Expected ComfyUI IMAGE tensor [B,H,W,C].")
    if int(images.shape[-1]) < 3:
        raise NativeNRInteropError("Native NR interop requires at least RGB input.")

    batch, in_h, in_w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    channels = 4 if int(images.shape[-1]) > 3 else 3
    out_w, out_h, factor = _target_size(in_w, in_h, upscale_mode)
    shape = (batch, out_h, out_w, channels)
    out_dtype = resolve_dtype(requested=output_precision, shape=shape, input_dtype=images.dtype)
    output_bytes = batch * out_h * out_w * channels * torch.empty((), dtype=out_dtype).element_size()

    if not torch.cuda.is_available() or not (0 <= int(gpu_index) < torch.cuda.device_count()):
        raise NativeNRInteropError(
            f"Requested CUDA device {gpu_index} is unavailable; PyTorch sees {torch.cuda.device_count()} CUDA device(s)."
        )
    cuda_device = torch.device(f"cuda:{int(gpu_index)}")

    vram_free_before_mb = legacy_nr._free_vram_mb(int(gpu_index))
    release_actions = ""
    if str(vram_guard) == "release_models" or (
        str(vram_guard) == "auto" and vram_free_before_mb < float(safety_margin_mb)
    ):
        release_actions = legacy_nr._release_comfy_models()
    vram_free_after_release_mb = legacy_nr._free_vram_mb(int(gpu_index))

    manifest: dict[str, Any]
    with _lock:
        lib, bridge_gpu, bridge_name, manifest = _ensure_session(
            int(gpu_index), runtime_path, bool(auto_bootstrap)
        )
        cuda_supported = bool(lib.dlss5nr_cuda_supported())
        cuda_status = _cuda_status(lib)

    gpu_direct = False
    out_full_gpu: torch.Tensor | None = None
    out_cpu: torch.Tensor | None = None
    storage: StorageInfo | None = None

    if output_device == "same_as_input" and images.device.type == "cuda" and images.device.index == int(gpu_index):
        try:
            free, _ = torch.cuda.mem_get_info(cuda_device)
            reserve = max(int(safety_margin_mb) * 1024 * 1024, 1024**3)
            if output_bytes + reserve <= int(free):
                out_full_gpu = torch.empty(shape, dtype=out_dtype, device=cuda_device)
                gpu_direct = True
        except Exception:
            pass

    if not gpu_direct:
        out_cpu, storage = allocate_cpu_tensor(
            shape,
            dtype=out_dtype,
            storage_mode=output_storage,
            prefix="nr_native",
            clean_cache=bool(clean_cache),
        )

    cuts = _scene_cut_flags(
        images,
        motion,
        enabled=(temporal_mode == "scene_cut_aware" and bool(reset_on_scene_cut)),
        threshold=float(scene_change_threshold),
    )

    style_i = _style_to_int(style)
    requested_chunk = max(1, int(chunk_frames))
    cuda_active = bool(cuda_supported)
    used_cuda = False
    used_cpu = False
    cuda_fallback_reason: str | None = None
    chunk_history: list[int] = []
    swap_key = (int(bridge_gpu), out_w, out_h)
    swap_decision = _channel_swap_cache.get(swap_key)

    progress = ConsoleProgress("Neural Rendering / native interop", batch, unit="frame")
    try:
        start = 0
        while start < batch:
            throw_if_interrupted()
            n = _choose_chunk_frames(
                requested_chunk,
                cuda_device=cuda_device,
                in_w=in_w,
                in_h=in_h,
                out_w=out_w,
                out_h=out_h,
                safety_margin_mb=int(safety_margin_mb),
                remaining=batch - start,
            )
            end = min(batch, start + n)
            chunk_history.append(end - start)

            # Keep transfer/resample scoped to the chunk. On CPU-only staging
            # fallback, we do not retain an extra GPU copy longer than needed.
            source_chunk = images[start:end, ..., :3].detach()
            if cuda_active:
                rgb = source_chunk.to(cuda_device, dtype=torch.float32, non_blocking=False).contiguous()
                if (out_w, out_h) != (in_w, in_h):
                    rgb = F.interpolate(
                        rgb.permute(0, 3, 1, 2),
                        size=(out_h, out_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).permute(0, 2, 3, 1).clamp_(0.0, 1.0).contiguous()
                chunk_out = torch.empty((end - start, out_h, out_w, 3), dtype=torch.float32, device=cuda_device)
            else:
                rgb = source_chunk.to(device="cpu", dtype=torch.float32).contiguous()
                if (out_w, out_h) != (in_w, in_h):
                    rgb = F.interpolate(
                        rgb.permute(0, 3, 1, 2),
                        size=(out_h, out_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).permute(0, 2, 3, 1).clamp_(0.0, 1.0).contiguous()
                chunk_out = torch.empty((end - start, out_h, out_w, 3), dtype=torch.float32, device="cpu")

            for local_i in range(end - start):
                throw_if_interrupted()
                global_i = start + local_i
                if temporal_mode == "still_images":
                    reset = 1
                elif temporal_mode == "temporal_sequence":
                    reset = 1 if global_i == 0 else 0
                else:
                    cut_before = global_i > 0 and (global_i - 1) < len(cuts) and cuts[global_i - 1]
                    reset = 1 if global_i == 0 or cut_before else 0

                if cuda_active:
                    try:
                        with _lock:
                            _process_cuda_frame(
                                lib,
                                rgb[local_i],
                                chunk_out[local_i],
                                style_i=style_i,
                                preset=int(preset),
                                intensity=float(intensity),
                                tone=float(tone),
                                structure=float(structure),
                                skin=float(skin),
                                auto_mask=bool(auto_mask),
                                reset=int(reset),
                            )
                        used_cuda = True
                    except NativeNRInteropError as exc:
                        cuda_active = False
                        cuda_fallback_reason = f"frame {global_i}: {exc}; {_cuda_status(lib)}"
                        print(
                            f"[AetherScale] Native NR CUDA interop fallback -> CPU staging: {cuda_fallback_reason}",
                            flush=True,
                        )
                        cpu_result = _process_cpu_frame(
                            lib,
                            rgb[local_i],
                            style_i=style_i,
                            preset=int(preset),
                            intensity=float(intensity),
                            tone=float(tone),
                            structure=float(structure),
                            skin=float(skin),
                            auto_mask=bool(auto_mask),
                            reset=int(reset),
                        )
                        chunk_out[local_i].copy_(cpu_result.to(chunk_out.device))
                        used_cpu = True
                else:
                    cpu_result = _process_cpu_frame(
                        lib,
                        rgb[local_i],
                        style_i=style_i,
                        preset=int(preset),
                        intensity=float(intensity),
                        tone=float(tone),
                        structure=float(structure),
                        skin=float(skin),
                        auto_mask=bool(auto_mask),
                        reset=int(reset),
                    )
                    chunk_out[local_i].copy_(cpu_result)
                    used_cpu = True
                progress.update(1)

            # Determine the stable runtime channel interpretation once per
            # resolution/session instead of synchronizing the GPU every frame.
            if channel_order == "BGRA":
                do_swap = True
            elif channel_order == "RGBA":
                do_swap = False
            else:
                if swap_decision is None:
                    ref0 = rgb[0]
                    raw0 = chunk_out[0]
                    if ref0.device != raw0.device:
                        ref0 = ref0.to(raw0.device)
                    swap_decision = _detect_channel_swap(ref0, raw0)
                    _channel_swap_cache[swap_key] = bool(swap_decision)
                do_swap = bool(swap_decision)
            if do_swap:
                chunk_out = chunk_out[..., [2, 1, 0]]
            chunk_out = chunk_out.clamp_(0.0, 1.0)

            if channels == 4:
                alpha = images[start:end, ..., 3:4].detach()
                if alpha.device != chunk_out.device:
                    alpha = alpha.to(chunk_out.device, dtype=torch.float32)
                else:
                    alpha = alpha.to(dtype=torch.float32)
                if (out_w, out_h) != (in_w, in_h):
                    alpha = F.interpolate(
                        alpha.permute(0, 3, 1, 2),
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                packed = torch.cat((chunk_out, alpha.clamp_(0.0, 1.0)), dim=-1)
            else:
                packed = chunk_out

            if gpu_direct:
                assert out_full_gpu is not None
                out_full_gpu[start:end].copy_(packed.to(dtype=out_dtype))
            else:
                assert out_cpu is not None
                cpu_chunk = packed.to(device="cpu", dtype=out_dtype, non_blocking=False)
                out_cpu[start:end].copy_(cpu_chunk)
                sync_file_backed_tensor(
                    out_cpu,
                    bytes_written=int(cpu_chunk.numel() * cpu_chunk.element_size()),
                )
                del cpu_chunk

            del source_chunk, rgb, chunk_out, packed
            start = end

        if out_cpu is not None:
            sync_file_backed_tensor(out_cpu, force=True)
    finally:
        progress.close(status="done" if start >= batch else "failed")

    result = out_full_gpu if gpu_direct else out_cpu
    assert result is not None

    stats: dict[str, Any] = {
        "engine": "DLSS 5 Neural Rendering / NGX feature 18",
        "backend": "native_interop",
        "bridge_origin": "third-party MIT dependency: orex2121/ComfyUI-DLSS5-orex",
        "bridge_commit": OREX_COMMIT,
        "integration_code": "AetherScale-owned ABI/storage/chunking adapter",
        "torch_cuda_device": int(gpu_index),
        "torch_gpu_name": torch.cuda.get_device_name(int(gpu_index)),
        "vram_guard": str(vram_guard),
        "vram_free_before_mb": round(vram_free_before_mb, 1),
        "vram_free_after_release_mb": round(vram_free_after_release_mb, 1),
        "release_actions": release_actions,
        "bridge_gpu_index": int(bridge_gpu),
        "bridge_gpu_name": bridge_name,
        "cuda_d3d12_interop_supported": bool(cuda_supported),
        "cuda_d3d12_interop_used": bool(used_cuda),
        "cpu_staging_used": bool(used_cpu),
        "cuda_status": cuda_status,
        "cuda_fallback_reason": cuda_fallback_reason,
        "frames": batch,
        "input_resolution": [in_w, in_h],
        "output_resolution": [out_w, out_h],
        "upscale_mode": str(upscale_mode),
        "upscale_factor": factor,
        "upscale_note": (
            "native resolution NR"
            if factor == 1.0
            else "GPU bicubic pre-resize + native NR; use AetherScale Super Resolution for true VSR"
        ),
        "chunk_frames_requested": int(chunk_frames),
        "chunk_frames_effective_min": min(chunk_history) if chunk_history else 0,
        "chunk_frames_effective_max": max(chunk_history) if chunk_history else 0,
        "scene_cuts_used": [i + 1 for i, flag in enumerate(cuts) if flag],
        "temporal_mode": temporal_mode,
        "style": style,
        "preset": int(preset),
        "intensity": float(intensity),
        "tone": float(tone),
        "structure": float(structure),
        "skin": float(skin),
        "auto_mask": bool(auto_mask),
        "channel_order": channel_order,
        "channel_swap_resolved": bool(swap_decision) if channel_order == "auto" and swap_decision is not None else None,
        "output_device": str(result.device),
        "output_precision": str(result.dtype).replace("torch.", ""),
        "clean_cache": bool(clean_cache),
        "vendor_manifest": manifest,
        **_storage_dict(storage, output_bytes=output_bytes, gpu_direct=gpu_direct),
    }
    return result, stats


def runtime_info() -> dict[str, Any]:
    bridge_ok = False
    caller_ok = False
    try:
        _verify_blob(
            VENDOR_BRIDGE,
            expected_size=OREX_BRIDGE_SIZE,
            expected_sha1=OREX_BRIDGE_BLOB_SHA1,
            label="OreX MIT bridge",
        )
        bridge_ok = True
    except Exception:
        pass
    try:
        _verify_blob(
            VENDOR_CALLER,
            expected_size=OREX_CALLER_SIZE,
            expected_sha1=OREX_CALLER_BLOB_SHA1,
            label="OreX MIT caller shim",
        )
        caller_ok = True
    except Exception:
        pass
    return {
        "backend": "native_interop",
        "bridge_ready": bridge_ok,
        "caller_ready": caller_ok,
        "runtime_ready": VENDOR_NR_DLL.is_file() or legacy_nr.NR_DLL.is_file(),
        "third_party_bridge_project": "orex2121/ComfyUI-DLSS5-orex",
        "third_party_commit": OREX_COMMIT,
        "integration_code": "independent AetherScale adapter; third-party binary used as a declared dependency",
        "zero_copy_capability": "CUDA/D3D12 external-memory path when bridge and selected GPU support it",
        "fallback": "CPU staging inside the same in-process bridge session",
    }
