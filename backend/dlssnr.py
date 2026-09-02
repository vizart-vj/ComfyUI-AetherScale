from __future__ import annotations

import ctypes
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .storage import allocate_cpu_tensor, resolve_dtype


ROOT = Path(__file__).resolve().parents[1]
NATIVE_BIN = ROOT / "native" / "bin"
RUNTIME_DIR = ROOT / "runtime" / "dlssnr"
CALLER_DIR = RUNTIME_DIR / "caller"
BRIDGE_DLL = NATIVE_BIN / "dlss5nr_bridge.dll"
CALLER_DLL = CALLER_DIR / "nvngx.dll_comfy.dll"
NR_DLL = RUNTIME_DIR / "nvngx_dlssnr.dll"

CORE_DLL = RUNTIME_DIR / "_nvngx.dll"
CORE_MANIFEST = RUNTIME_DIR / "ngx_core_manifest.json"
MIN_STOCK_DRIVER = (616, 56)

UPSTREAM_RELEASE = "v0.2.0"
UPSTREAM_ARCHIVE_URL = (
    "https://github.com/lisitskyaa/ComfyUI-DLSS5-NR/releases/download/"
    "v0.2.0/ComfyUI-DLSS5-NR-v0.2.0-windows-x64.zip"
)
UPSTREAM_ARCHIVE_SHA256 = "d10d6cd4e7b9d15ef43501baeff1c9fd7b5e3fe41a908b44c338813a82541260"

# Upstream v0.2.0 bridge opens the core session with engineVersion "0.2.0".
# The independently working dlss5-nr-player reference documents the observed
# NR identity contract as project 53f803cc-... + CUSTOM + engineVersion "0.1".
# The exact null-terminated literal is patchable in-place without changing PE size.
BRIDGE_ENGINE_OLD = b"0.2.0\x00"
BRIDGE_ENGINE_NEW = b"0.1\x00\x00\x00"
BRIDGE_PATCH_MANIFEST = NATIVE_BIN / "bridge_patch_manifest.json"


RHI_DLSSNR_BUILDS = {
    # RTX 50: stock signed/pre-release NR runtime.
    "rtx50": {
        "tag": "dlssnr-310.8.0",
        "url": "https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.0/nvngx_dlssnr_310.8.0.zip",
        "archive_sha256": "388c0a7912e15ec911b9c9e11a692142b11fe387ddf2b637d8c358138fffb3ac",
    },
    # Community-published RTX 40 compatible variant.
    "rtx40": {
        "tag": "dlssnr-310.8.0-RTX40",
        "url": "https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.0-RTX40/nvngx_dlssnr_310.8.0-RTX40.zip",
        "archive_sha256": "46124cfaef532ad5f6da07494772ea8c1b3e719f934e254385697f38d1289e3f",
    },
    # Broad fallback used by current RHI one-click tooling.
    "fallback": {
        "tag": "dlssnr-310.8.SF-v2",
        "url": "https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.SF-v2/nvngx_dlssnr_310.8.SF-v2.zip",
        "archive_sha256": "1da35941894994eb087e017577829e492454e9bae3a6a9397027069ceb74955c",
    },
}
RUNTIME_MANIFEST = RUNTIME_DIR / "runtime_manifest.json"

_lock = threading.RLock()
_lib: Optional[Any] = None
_initialized_gpu: Optional[int] = None
_dll_handles: list[Any] = []


class DLSSNRError(RuntimeError):
    pass


@dataclass(slots=True)
class DLSSNRRuntimeState:
    bridge_ready: bool
    caller_ready: bool
    runtime_ready: bool
    runtime_path: Optional[str]
    bridge_path: str
    caller_path: str
    message: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _decode_error(buf: ctypes.Array) -> str:
    try:
        return buf.value.decode("utf-8", errors="replace")
    except Exception:
        return "Unknown native DLSS5 NR error"


def _download(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-AetherScale/0.4.6"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_to_file(url: str, destination: Path, timeout: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-AetherScale/0.4.6"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp, destination.open("wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _gpu_generation(gpu_index: int = 0) -> str:
    try:
        if torch.cuda.is_available() and 0 <= gpu_index < torch.cuda.device_count():
            name = torch.cuda.get_device_name(gpu_index).upper()
            if "RTX 50" in name:
                return "rtx50"
            if "RTX 40" in name:
                return "rtx40"
    except Exception:
        pass
    return "fallback"


def _bootstrap_rhi_runtime(gpu_index: int = 0) -> Dict[str, Any]:
    profile_name = _gpu_generation(gpu_index)
    profile = RHI_DLSSNR_BUILDS[profile_name]

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = RUNTIME_DIR / f".{profile['tag']}.download.zip"

    try:
        print(
            f"[AetherScale] DLSSNR runtime not found locally; downloading "
            f"{profile['tag']} for profile={profile_name}..."
        )
        _download_to_file(profile["url"], tmp_zip)
        digest = _sha256_file(tmp_zip)
        expected = profile["archive_sha256"].lower()
        if digest != expected:
            raise DLSSNRError(
                "DLSSNR runtime archive checksum mismatch. "
                f"Expected {expected}, got {digest}."
            )

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            members = zf.namelist()
            dll_member = next(
                (
                    m for m in members
                    if Path(m).name.lower() == "nvngx_dlssnr.dll"
                ),
                None,
            )
            if dll_member is None:
                raise DLSSNRError(
                    f"RHI release {profile['tag']} does not contain nvngx_dlssnr.dll."
                )
            data = zf.read(dll_member)
            NR_DLL.write_bytes(data)

        manifest = {
            "source": "RankFTW/rhi-repo",
            "tag": profile["tag"],
            "profile": profile_name,
            "archive_url": profile["url"],
            "archive_sha256": digest,
            "runtime_sha256": _sha256_file(NR_DLL),
            "runtime_bytes": NR_DLL.stat().st_size,
        }
        RUNTIME_MANIFEST.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(
            f"[AetherScale] DLSSNR runtime ready: {profile['tag']} "
            f"({manifest['runtime_bytes'] / (1024**2):.1f} MiB)."
        )
        return manifest
    except DLSSNRError:
        if NR_DLL.exists():
            try:
                NR_DLL.unlink()
            except OSError:
                pass
        raise
    except Exception as exc:
        if NR_DLL.exists():
            try:
                NR_DLL.unlink()
            except OSError:
                pass
        raise DLSSNRError(
            f"Automatic DLSSNR runtime bootstrap from RankFTW/rhi-repo failed: {exc}"
        ) from exc
    finally:
        try:
            tmp_zip.unlink(missing_ok=True)
        except OSError:
            pass


def _patch_bridge_identity(path: Path) -> Dict[str, Any]:
    """Patch only the exact engineVersion literal in the MIT bridge binary.

    This is an in-place fixed-length data patch:
      b"0.2.0\\0" -> b"0.1\\0\\0\\0"

    No code bytes, section sizes, offsets, imports or relocations change.
    """
    if not path.is_file():
        raise DLSSNRError(f"Bridge binary is missing: {path}")

    data = path.read_bytes()
    old_count = data.count(BRIDGE_ENGINE_OLD)
    new_count = data.count(BRIDGE_ENGINE_NEW)

    before_hash = _sha256_bytes(data)

    if old_count == 1:
        patched = data.replace(BRIDGE_ENGINE_OLD, BRIDGE_ENGINE_NEW, 1)
        if len(patched) != len(data):
            raise DLSSNRError("Internal bridge patch changed binary size; refusing to write.")
        path.write_bytes(patched)
        after_hash = _sha256_bytes(patched)
        status = "patched"
    elif old_count == 0 and new_count >= 1:
        after_hash = before_hash
        status = "already_patched"
    else:
        raise DLSSNRError(
            "Unexpected DLSS5 bridge identity layout. "
            f"Found old engineVersion literal {old_count} times and patched literal {new_count} times. "
            "Refusing to patch an unknown binary."
        )

    manifest = {
        "status": status,
        "path": str(path),
        "source_release": UPSTREAM_RELEASE,
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "old_literal_matches": old_count,
        "new_literal_matches": data.count(BRIDGE_ENGINE_NEW) if status == "already_patched" else patched.count(BRIDGE_ENGINE_NEW),
        "project_id": "53f803cc-a12f-4d69-90d5-19b7599cad19",
        "engine_type": "CUSTOM",
        "engine_version": "0.1",
        "reason": (
            "Feature 18 requires an Init_ProjectID session using the observed "
            "DLSSNR identity contract; plain/fallback Init_Ext sessions are rejected."
        ),
    }
    try:
        BRIDGE_PATCH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        BRIDGE_PATCH_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        pass

    if status == "patched":
        print(
            "[AetherScale] DLSS5 bridge identity patched: "
            'engineVersion "0.2.0" -> "0.1" (fixed-length PE data patch).'
        )
    return manifest


def ensure_bridge(force: bool = False) -> Dict[str, Any]:
    """Install only the MIT project-owned bridge/caller helper.

    No NVIDIA proprietary runtime is downloaded or redistributed here.
    """
    with _lock:
        if BRIDGE_DLL.is_file() and CALLER_DLL.is_file() and not force:
            patch = _patch_bridge_identity(BRIDGE_DLL)
            return {
                "ready": True,
                "bridge": str(BRIDGE_DLL),
                "caller": str(CALLER_DLL),
                "source": f"lisitskyaa/ComfyUI-DLSS5-NR {UPSTREAM_RELEASE}",
                "identity_patch": patch,
            }

        NATIVE_BIN.mkdir(parents=True, exist_ok=True)
        CALLER_DIR.mkdir(parents=True, exist_ok=True)
        archive = _download(UPSTREAM_ARCHIVE_URL)
        digest = _sha256_bytes(archive)
        if digest != UPSTREAM_ARCHIVE_SHA256:
            raise DLSSNRError(
                "DLSS5 bridge bootstrap checksum mismatch. "
                f"Expected {UPSTREAM_ARCHIVE_SHA256}, got {digest}."
            )

        with zipfile.ZipFile(io.BytesIO(archive), "r") as zf:
            members = zf.namelist()
            bridge_member = next((m for m in members if m.endswith("native/bin/dlss5nr_bridge.dll")), None)
            caller_member = next((m for m in members if m.endswith("runtime/caller/nvngx.dll_comfy.dll")), None)
            if bridge_member is None or caller_member is None:
                raise DLSSNRError("Upstream DLSS5 release layout changed; bridge/caller files were not found.")
            BRIDGE_DLL.write_bytes(zf.read(bridge_member))
            CALLER_DLL.write_bytes(zf.read(caller_member))

        patch = _patch_bridge_identity(BRIDGE_DLL)
        return {
            "ready": True,
            "bridge": str(BRIDGE_DLL),
            "caller": str(CALLER_DLL),
            "source": f"lisitskyaa/ComfyUI-DLSS5-NR {UPSTREAM_RELEASE}",
            "archive_sha256": digest,
            "identity_patch": patch,
        }


def _bounded_find_named(root: Path, filename: str, *, max_depth: int = 4) -> list[Path]:
    """Find filename under root without recursively crawling an entire disk.

    Used for NVIDIA's own install/cache roots only. DriverStore can contain many
    packages, so depth is capped and permission errors are ignored.
    """
    out: list[Path] = []
    if not root.is_dir():
        return out
    root_parts = len(root.parts)
    try:
        for current, dirs, files in os.walk(root):
            cur = Path(current)
            depth = len(cur.parts) - root_parts
            if depth >= max_depth:
                dirs[:] = []
            if filename.lower() in {f.lower() for f in files}:
                for f in files:
                    if f.lower() == filename.lower():
                        out.append(cur / f)
    except OSError:
        pass
    return out


def _nvidia_system_runtime_candidates() -> list[Path]:
    """Discover an NVIDIA-provided DLSSNR runtime already installed on Windows.

    We do not download NVIDIA proprietary binaries. Instead we automatically
    stage an authentic local copy shipped by the installed NVIDIA driver/app
    when present.
    """
    if os.name != "nt":
        return []

    out: list[Path] = []

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    driver_store = system_root / "System32" / "DriverStore" / "FileRepository"

    # NVIDIA driver packages are top-level nv*.inf_* directories. Search only
    # those packages and only a shallow depth.
    if driver_store.is_dir():
        try:
            packages = sorted(
                (p for p in driver_store.glob("nv*.inf_*") if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            packages = []

        for pkg in packages:
            direct = pkg / "nvngx_dlssnr.dll"
            if direct.is_file():
                out.append(direct)
            out.extend(_bounded_find_named(pkg, "nvngx_dlssnr.dll", max_depth=3))

    # NVIDIA App / driver installer caches. Different driver generations place
    # extracted packages under different ProgramData directories.
    program_data = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
    for root in (
        program_data / "NVIDIA Corporation",
        program_data / "NVIDIA",
        program_data / "NVIDIA Corporation" / "Downloader",
        program_data / "NVIDIA Corporation" / "NVIDIA App",
    ):
        out.extend(_bounded_find_named(root, "nvngx_dlssnr.dll", max_depth=5))

    # Standard NVIDIA install roots can also contain NGX components.
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if not base:
            continue
        for root in (
            Path(base) / "NVIDIA Corporation",
            Path(base) / "NVIDIA",
        ):
            out.extend(_bounded_find_named(root, "nvngx_dlssnr.dll", max_depth=5))

    # De-duplicate and prefer newest local NVIDIA copy.
    unique: dict[str, Path] = {}
    for p in out:
        try:
            unique[str(p.resolve()).lower()] = p
        except OSError:
            unique[str(p).lower()] = p

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(unique.values(), key=mtime, reverse=True)

def _candidate_runtime_paths(custom_path: str = "") -> list[Path]:
    candidates: list[Path] = []
    if custom_path.strip():
        p = Path(os.path.expandvars(os.path.expanduser(custom_path.strip())))
        candidates.append(p if p.suffix.lower() == ".dll" else p / "nvngx_dlssnr.dll")

    candidates.append(NR_DLL)

    # Zero-action path: use the NVIDIA runtime already installed by the display
    # driver / NVIDIA App, then stage a private copy into AetherScale.
    candidates.extend(_nvidia_system_runtime_candidates())

    # Sibling/installed ComfyUI node runtime.
    custom_nodes = ROOT.parent
    candidates.append(custom_nodes / "ComfyUI-DLSS5-NR" / "runtime" / "nvngx_dlssnr.dll")

    # Common RHI local cache roots. Scan only these bounded locations.
    local = os.environ.get("LOCALAPPDATA")
    if local:
        rhi_root = Path(local) / "RHI"
        for direct in (
            rhi_root / "nvngx_dlssnr.dll",
            rhi_root / "rdx5" / "nvngx_dlssnr.dll",
            rhi_root / "runtime" / "nvngx_dlssnr.dll",
            rhi_root / "dlss" / "nvngx_dlssnr.dll",
        ):
            candidates.append(direct)
        if rhi_root.is_dir():
            try:
                # Bounded by RHI directory only; first matching runtime wins.
                candidates.extend(rhi_root.rglob("nvngx_dlssnr.dll"))
            except OSError:
                pass

    # De-duplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def discover_runtime(custom_path: str = "", stage: bool = True) -> Optional[Path]:
    for candidate in _candidate_runtime_paths(custom_path):
        if candidate.is_file() and candidate.name.lower() == "nvngx_dlssnr.dll":
            if stage and candidate.resolve() != NR_DLL.resolve():
                RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(candidate, NR_DLL)
                    return NR_DLL
                except OSError as exc:
                    raise DLSSNRError(
                        f"Found DLSSNR runtime at {candidate}, but could not stage it into AetherScale: {exc}"
                    ) from exc
            return candidate
    return None


def probe(custom_path: str = "") -> DLSSNRRuntimeState:
    runtime = discover_runtime(custom_path=custom_path, stage=False)
    ready = BRIDGE_DLL.is_file() and CALLER_DLL.is_file() and runtime is not None
    if ready:
        msg = "DLSS5 Neural Rendering bridge and runtime are ready."
    elif runtime is None:
        msg = "Compatible nvngx_dlssnr.dll was not found locally."
    else:
        msg = "NVIDIA runtime found, but the MIT bridge/caller helper is not bootstrapped yet."
    return DLSSNRRuntimeState(
        bridge_ready=BRIDGE_DLL.is_file(),
        caller_ready=CALLER_DLL.is_file(),
        runtime_ready=runtime is not None,
        runtime_path=str(runtime) if runtime else None,
        bridge_path=str(BRIDGE_DLL),
        caller_path=str(CALLER_DLL),
        message=msg,
    )


def _parse_driver_version(value: str) -> tuple[int, int] | None:
    m = re.search(r"(\d+)\.(\d+)", value or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _driver_version(gpu_index: int = 0) -> dict[str, Any]:
    """Return the active NVIDIA display-driver version, not a CUDA toolkit version."""
    info: dict[str, Any] = {
        "raw": None,
        "parsed": None,
        "source": None,
    }
    if os.name != "nt":
        return info

    # nvidia-smi is installed with the Windows display driver and is the least
    # ambiguous source for the actually active kernel/display driver.
    candidates = [
        "nvidia-smi.exe",
        str(Path(os.environ.get("ProgramW6432", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"),
        str(Path(os.environ.get("SystemRoot", r"C:\Windows"))
            / "System32" / "nvidia-smi.exe"),
    ]
    for exe in candidates:
        try:
            cp = subprocess.run(
                [
                    exe,
                    f"--id={int(gpu_index)}",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            raw = (cp.stdout or "").strip().splitlines()
            if cp.returncode == 0 and raw:
                version = raw[0].strip()
                parsed = _parse_driver_version(version)
                info.update(
                    raw=version,
                    parsed=list(parsed) if parsed else None,
                    source="nvidia-smi",
                )
                return info
        except Exception:
            continue
    return info


def _win_file_version(path: Path) -> tuple[int, int, int, int] | None:
    if os.name != "nt" or not path.is_file():
        return None
    try:
        version = ctypes.WinDLL("version.dll", use_last_error=True)
        size = version.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(str(path), 0, size, buf):
            return None

        lp = ctypes.c_void_p()
        ln = ctypes.c_uint()
        if not version.VerQueryValueW(buf, "\\", ctypes.byref(lp), ctypes.byref(ln)):
            return None

        class VS_FIXEDFILEINFO(ctypes.Structure):
            _fields_ = [
                ("dwSignature", ctypes.c_uint32),
                ("dwStrucVersion", ctypes.c_uint32),
                ("dwFileVersionMS", ctypes.c_uint32),
                ("dwFileVersionLS", ctypes.c_uint32),
                ("dwProductVersionMS", ctypes.c_uint32),
                ("dwProductVersionLS", ctypes.c_uint32),
                ("dwFileFlagsMask", ctypes.c_uint32),
                ("dwFileFlags", ctypes.c_uint32),
                ("dwFileOS", ctypes.c_uint32),
                ("dwFileType", ctypes.c_uint32),
                ("dwFileSubtype", ctypes.c_uint32),
                ("dwFileDateMS", ctypes.c_uint32),
                ("dwFileDateLS", ctypes.c_uint32),
            ]

        ffi = ctypes.cast(lp, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
        return (
            (ffi.dwFileVersionMS >> 16) & 0xFFFF,
            ffi.dwFileVersionMS & 0xFFFF,
            (ffi.dwFileVersionLS >> 16) & 0xFFFF,
            ffi.dwFileVersionLS & 0xFFFF,
        )
    except Exception:
        return None


def _ngx_core_candidates() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    repo = system_root / "System32" / "DriverStore" / "FileRepository"
    out: list[dict[str, Any]] = []

    if repo.is_dir():
        try:
            packages = list(repo.glob("nv*.inf_*"))
        except OSError:
            packages = []

        for package in packages:
            p = package / "_nvngx.dll"
            if not p.is_file():
                continue
            ver = _win_file_version(p)
            try:
                stamp = p.stat().st_mtime
            except OSError:
                stamp = 0.0
            out.append(
                {
                    "path": p,
                    "version": ver,
                    "mtime": stamp,
                    "package": package.name,
                }
            )

    # Highest file version first. This is intentionally NOT mtime-first:
    # DriverStore can retain old packages with newer timestamps after repair/update.
    out.sort(
        key=lambda x: (
            x["version"] if x["version"] is not None else (0, 0, 0, 0),
            x["mtime"],
        ),
        reverse=True,
    )
    return out


def _stage_best_ngx_core(gpu_index: int = 0) -> dict[str, Any]:
    """Stage the newest NGX core into runtime/_nvngx.dll.

    The native bridge prioritizes this exact local override, preventing it from
    accidentally loading a stale retained DriverStore package.
    """
    candidates = _ngx_core_candidates()
    if not candidates:
        return {
            "ready": CORE_DLL.is_file(),
            "source": str(CORE_DLL) if CORE_DLL.is_file() else None,
            "reason": "no_driverstore_core_candidates",
        }

    best = candidates[0]
    src: Path = best["path"]

    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        src_hash = _sha256_file(src)
        dst_hash = _sha256_file(CORE_DLL) if CORE_DLL.is_file() else None
        if src_hash != dst_hash:
            shutil.copy2(src, CORE_DLL)

        manifest = {
            "source": str(src),
            "source_package": best["package"],
            "file_version": list(best["version"]) if best["version"] else None,
            "sha256": src_hash,
            "candidate_count": len(candidates),
            "driver": _driver_version(gpu_index),
        }
        CORE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            "[AetherScale] NGX core staged: "
            f"{src.name} version={manifest['file_version']} "
            f"package={best['package']}"
        )
        return {"ready": True, **manifest}
    except Exception as exc:
        return {
            "ready": CORE_DLL.is_file(),
            "source": str(src),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _compatibility_preflight(gpu_index: int = 0) -> dict[str, Any]:
    gpu_name = "unknown"
    try:
        if torch.cuda.is_available() and 0 <= gpu_index < torch.cuda.device_count():
            gpu_name = torch.cuda.get_device_name(gpu_index)
    except Exception:
        pass

    generation = _gpu_generation(gpu_index)
    driver = _driver_version(gpu_index)
    parsed = tuple(driver["parsed"]) if driver.get("parsed") else None
    stock_driver_ok = parsed is None or parsed >= MIN_STOCK_DRIVER

    core = _stage_best_ngx_core(gpu_index)
    return {
        "gpu_name": gpu_name,
        "gpu_generation": generation,
        "driver": driver,
        "minimum_stock_driver": f"{MIN_STOCK_DRIVER[0]}.{MIN_STOCK_DRIVER[1]:02d}",
        "stock_driver_ok": stock_driver_ok,
        "ngx_core": core,
    }


def ensure_runtime(*, custom_path: str = "", auto_bootstrap: bool = True, gpu_index: int = 0) -> DLSSNRRuntimeState:
    if platform.system() != "Windows":
        raise DLSSNRError("DLSS 5 Neural Rendering requires Windows/D3D12.")
    if auto_bootstrap:
        ensure_bridge(force=False)

    preflight = _compatibility_preflight(gpu_index=int(gpu_index))
    runtime = discover_runtime(custom_path=custom_path, stage=True)
    if runtime is None and auto_bootstrap:
        # True zero-action install: if the NVIDIA driver does not already carry
        # DLSSNR, fetch the matching RHI-published runtime automatically.
        _bootstrap_rhi_runtime(gpu_index=int(gpu_index))
        runtime = discover_runtime(custom_path="", stage=True)

    if runtime is None:
        raise DLSSNRError(
            "DLSSNR runtime is missing and automatic bootstrap is disabled or failed. "
            "Enable auto_bootstrap to let AetherScale download and stage the appropriate "
            "RankFTW/rhi-repo DLSSNR release automatically."
        )

    # Stock Blackwell runtime currently needs an NGX/display-driver stack at
    # least as new as 616.56. Fail before processing a long video.
    if (
        preflight["gpu_generation"] == "rtx50"
        and not preflight["stock_driver_ok"]
    ):
        drv = preflight["driver"].get("raw") or "unknown"
        raise DLSSNRError(
            "DLSS 5 NR compatibility preflight failed: "
            f"active NVIDIA driver is {drv}, while the current stock DLSSNR 310.8.0 "
            f"path requires driver >= {preflight['minimum_stock_driver']}. "
            "0xBAD00001 is NGX FAIL_FeatureNotSupported on this stack. "
            "AetherScale has already staged the newest matching _nvngx.dll available "
            "from your installed DriverStore; a newer display driver is required."
        )

    state = probe(custom_path="")
    if not state.bridge_ready or not state.caller_ready:
        raise DLSSNRError("DLSS5 bridge bootstrap is incomplete.")
    return state


def _register_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for p in (NATIVE_BIN, RUNTIME_DIR, CALLER_DIR):
        if p.is_dir():
            try:
                _dll_handles.append(os.add_dll_directory(str(p)))
            except OSError:
                pass


def _load_library() -> Any:
    global _lib
    if _lib is not None:
        return _lib
    if platform.system() != "Windows":
        raise DLSSNRError("DLSS 5 Neural Rendering requires Windows/D3D12.")
    if not BRIDGE_DLL.is_file():
        raise DLSSNRError(f"Native bridge is missing: {BRIDGE_DLL}")

    _register_dll_dirs()
    lib = ctypes.WinDLL(str(BRIDGE_DLL))
    lib.dlss5nr_init.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int]
    lib.dlss5nr_init.restype = ctypes.c_int
    lib.dlss5nr_process.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.dlss5nr_process.restype = ctypes.c_int
    lib.dlss5nr_shutdown.argtypes = []
    lib.dlss5nr_shutdown.restype = None
    lib.dlss5nr_version.argtypes = []
    lib.dlss5nr_version.restype = ctypes.c_char_p
    lib.dlss5nr_gpu_name.argtypes = []
    lib.dlss5nr_gpu_name.restype = ctypes.c_char_p
    _lib = lib
    return lib


def shutdown() -> None:
    global _initialized_gpu
    with _lock:
        if _lib is not None and _initialized_gpu is not None:
            try:
                _lib.dlss5nr_shutdown()
            except Exception:
                pass
        _initialized_gpu = None


def _ensure_initialized(gpu_index: int, custom_path: str, auto_bootstrap: bool) -> Any:
    global _initialized_gpu
    ensure_runtime(custom_path=custom_path, auto_bootstrap=auto_bootstrap, gpu_index=int(gpu_index))
    lib = _load_library()
    if _initialized_gpu == gpu_index:
        return lib
    if _initialized_gpu is not None:
        lib.dlss5nr_shutdown()
        _initialized_gpu = None

    err = ctypes.create_string_buffer(4096)
    ok = lib.dlss5nr_init(int(gpu_index), str(RUNTIME_DIR), err, len(err))
    if not ok:
        raise DLSSNRError(_decode_error(err))
    _initialized_gpu = int(gpu_index)
    return lib


def _release_comfy_models() -> str:
    actions: list[str] = []
    try:
        import comfy.model_management as mm
        unload = getattr(mm, "unload_all_models", None)
        if callable(unload):
            unload()
            actions.append("unload_all_models")
        soft = getattr(mm, "soft_empty_cache", None)
        if callable(soft):
            try:
                soft()
            except TypeError:
                soft(force=True)
            actions.append("soft_empty_cache")
    except Exception as exc:
        actions.append(f"comfy_release_unavailable:{type(exc).__name__}")
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            actions.append("torch_empty_cache")
        except Exception:
            pass
    return ",".join(actions)


def _free_vram_mb(device: int) -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_b, _ = torch.cuda.mem_get_info(device)
    except TypeError:
        with torch.cuda.device(device):
            free_b, _ = torch.cuda.mem_get_info()
    return float(free_b) / (1024.0 * 1024.0)


def _style_to_int(style: str) -> int:
    named = {"default": 0, "natural": 1, "cinematic": 2}
    if style in named:
        return named[style]
    return int(style)


def _channel_correct(frame_out: np.ndarray, frame_in: np.ndarray, mode: str) -> np.ndarray:
    if mode == "RGBA":
        return frame_out
    if mode == "BGRA":
        return frame_out[..., [2, 1, 0]]
    h, w, _ = frame_in.shape
    raw = frame_out
    swapped = frame_out[..., [2, 1, 0]]
    step_y = max(1, h // 128)
    step_x = max(1, w // 128)
    ref_s = frame_in[::step_y, ::step_x]
    raw_s = raw[::step_y, ::step_x]
    swp_s = swapped[::step_y, ::step_x]
    raw_score = float(np.mean(np.abs(raw_s - ref_s))) + float(
        np.mean(np.abs(raw_s.mean(axis=(0, 1)) - ref_s.mean(axis=(0, 1))))
    )
    swp_score = float(np.mean(np.abs(swp_s - ref_s))) + float(
        np.mean(np.abs(swp_s.mean(axis=(0, 1)) - ref_s.mean(axis=(0, 1))))
    )
    return raw if raw_score <= swp_score else swapped


def process(
    images: torch.Tensor,
    *,
    style: str,
    preset: int,
    intensity: float,
    tone: float,
    structure: float,
    skin: float,
    auto_mask: bool,
    temporal_mode: str,
    channel_order: str,
    gpu_index: int,
    vram_guard: str,
    min_free_vram_mb: int,
    output_device: str,
    auto_bootstrap: bool,
    runtime_path: str = "",
    motion: Optional[Any] = None,
    output_precision: str = "auto",
    output_storage: str = "auto",
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise DLSSNRError("Expected ComfyUI IMAGE tensor [B,H,W,C].")
    if images.shape[-1] < 3:
        raise DLSSNRError("DLSS5 NR requires at least RGB input.")

    before_free=_free_vram_mb(int(gpu_index))
    release_actions=""
    if vram_guard=="release_models" or (vram_guard=="auto" and before_free<float(min_free_vram_mb)):
        release_actions=_release_comfy_models()
    after_release=_free_vram_mb(int(gpu_index))

    style_i=_style_to_int(style)
    batch,h,w=int(images.shape[0]),int(images.shape[1]),int(images.shape[2])
    channels=4 if int(images.shape[-1])>3 else 3
    shape=(batch,h,w,channels)
    out_dtype=resolve_dtype(
        requested=output_precision,
        shape=shape,
        input_dtype=images.dtype,
    )
    out_cpu, storage=allocate_cpu_tensor(
        shape,dtype=out_dtype,storage_mode=output_storage,prefix="dlssnr"
    )

    scene_cuts=[]
    if motion is not None:
        try: scene_cuts=[bool(x) for x in motion.scene_cuts.detach().cpu().tolist()]
        except Exception: scene_cuts=[]

    # Working set is bounded to one frame:
    # input frame float32 + native D3D12 resources + output frame float32.
    with _lock:
        lib=_ensure_initialized(int(gpu_index),runtime_path,bool(auto_bootstrap))
        for i in range(batch):
            frame_t=images[i,...,:3].detach().to(device="cpu",dtype=torch.float32).contiguous()
            frame_in=frame_t.numpy()
            frame_out=np.empty((h,w,3),dtype=np.float32)

            if temporal_mode=="still_images":
                reset=1
            elif temporal_mode=="temporal_sequence":
                reset=1 if i==0 else 0
            else:
                cut_before=i>0 and (i-1)<len(scene_cuts) and scene_cuts[i-1]
                reset=1 if (i==0 or cut_before) else 0

            err=ctypes.create_string_buffer(4096)
            ok=lib.dlss5nr_process(
                frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w,h,int(style_i),int(preset),
                ctypes.c_float(float(intensity)),ctypes.c_float(float(tone)),
                ctypes.c_float(float(structure)),ctypes.c_float(float(skin)),
                1 if auto_mask else 0,int(reset),err,len(err)
            )
            if not ok:
                native_error = _decode_error(err)
                if "BAD00001" in native_error.upper():
                    pf = _compatibility_preflight(int(gpu_index))
                    drv = pf["driver"].get("raw") or "unknown"
                    core_ver = pf["ngx_core"].get("file_version")
                    raise DLSSNRError(
                        f"Frame {i}: {native_error} "
                        f"Compatibility: GPU={pf['gpu_name']}; driver={drv}; "
                        f"staged _nvngx.dll version={core_ver}; "
                        f"minimum verified stock driver={pf['minimum_stock_driver']}. "
                        "BAD00001 means NGX rejected feature 18 as unsupported even after the "
                        "ProjectID/CUSTOM/engineVersion=0.1 bridge identity patch; "
                        "this is not a VRAM, image-size, or caller-shim failure."
                    )
                raise DLSSNRError(f"Frame {i}: {native_error}")
            corrected=np.ascontiguousarray(_channel_correct(frame_out,frame_in,channel_order))
            out_cpu[i,...,:3].copy_(torch.from_numpy(corrected).to(dtype=out_dtype))
            if channels==4:
                # No full-batch alpha clone.
                out_cpu[i,...,3:4].copy_(
                    images[i,...,3:4].detach().to(device="cpu",dtype=out_dtype)
                )
            del frame_t,frame_in,frame_out,corrected

    out_cpu.clamp_(0,1)
    if output_device=="same_as_input" and images.device.type!="cpu":
        result=out_cpu.to(images.device,non_blocking=False)
    else:
        # Critical: do NOT cast the complete long-video output back to input dtype.
        result=out_cpu

    bridge_version=gpu_name="unknown"
    lib=_lib
    if lib is not None:
        try:
            v=lib.dlss5nr_version(); bridge_version=v.decode("utf-8",errors="replace") if v else "unknown"
        except Exception: pass
        try:
            g=lib.dlss5nr_gpu_name(); gpu_name=g.decode("utf-8",errors="replace") if g else "unknown"
        except Exception: pass

    final_free=_free_vram_mb(int(gpu_index))
    state=probe()
    stats={
        "engine":"DLSS 5 Neural Rendering / NGX feature 18",
        "bridge_version":bridge_version,"gpu":gpu_name,"gpu_index":int(gpu_index),
        "style":style,"preset":int(preset),"intensity":float(intensity),
        "tone":float(tone),"structure":float(structure),"skin":float(skin),
        "auto_mask":bool(auto_mask),"temporal_mode":temporal_mode,
        "scene_cuts_used":scene_cuts if temporal_mode=="scene_cut_aware" else [],
        "channel_order":channel_order,"frames":batch,"resolution":[w,h],
        "output_device":str(result.device),
        "output_precision":storage.dtype,
        "output_storage_backend":storage.backend,
        "output_storage_path":storage.path,
        "output_gib":round(storage.bytes/1024**3,3),
        "per_frame_cpu_staging":True,
        "full_batch_cpu_output_allocation":storage.backend=="ram",
        "cuda_d3d12_zero_copy":False,
        "vram_guard":vram_guard,
        "vram_free_before_mb":round(before_free,1),
        "vram_free_after_release_mb":round(after_release,1),
        "vram_free_final_mb":round(final_free,1),
        "release_actions":release_actions,
        "runtime":state.runtime_path,
        "runtime_sha256":_sha256_file(NR_DLL) if NR_DLL.is_file() else None,
        "reference_bridge":"lisitskyaa/ComfyUI-DLSS5-NR v0.2.0 (MIT)",
    }
    return result,stats

def runtime_info(custom_path: str = "") -> Dict[str, Any]:
    state = probe(custom_path=custom_path)
    return {
        "bridge_ready": state.bridge_ready,
        "caller_ready": state.caller_ready,
        "runtime_ready": state.runtime_ready,
        "runtime_path": state.runtime_path,
        "bridge_path": state.bridge_path,
        "caller_path": state.caller_path,
        "message": state.message,
        "bridge_bootstrap_source": f"lisitskyaa/ComfyUI-DLSS5-NR {UPSTREAM_RELEASE}",
        "feature_id": 18,
        "cpu_staging": True,
        "zero_copy_status": "planned",
    }
