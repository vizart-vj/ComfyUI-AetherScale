from __future__ import annotations

from dataclasses import dataclass
import mmap as py_mmap
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Sequence
import weakref

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CACHE_DIR = (ROOT / ".aetherscale_cache").resolve()

def _default_cache_dir() -> Path:
    configured = os.environ.get("AETHERSCALE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        import folder_paths  # type: ignore
        return (Path(folder_paths.get_temp_directory()) / "aetherscale_cache").resolve()
    except Exception:
        return LEGACY_CACHE_DIR

CACHE_DIR = _default_cache_dir()
_CACHE_LOCK = threading.RLock()
_ACTIVE_PATHS: set[str] = set()
_PENDING_DELETE: set[str] = set()
_MMAP_COUNTER = 0


def _spill_flush_threshold_bytes() -> int:
    raw = str(os.environ.get("AETHERSCALE_SPILL_FLUSH_MB", "128")).strip()
    try:
        mb = max(16, int(float(raw)))
    except Exception:
        mb = 128
    return mb * 1024 * 1024



def sync_file_backed_tensor(
    tensor: torch.Tensor,
    *,
    bytes_written: int = 0,
    force: bool = False,
) -> bool:
    """Bound dirty-page writeback for AetherScale file-backed outputs.

    Old spill mappings were allowed to accumulate tens of GiB of dirty mapped
    pages. If ComfyUI was killed, the Windows Memory Manager kept flushing those
    pages to disk after Python had already exited, which looked like a zombie
    ComfyUI workload. We now impose writeback backpressure every bounded chunk.
    Returns True when a flush occurred.
    """
    mm = getattr(tensor, "_aetherscale_memmap", None)
    if mm is None:
        return False
    pending = int(getattr(tensor, "_aetherscale_dirty_bytes", 0)) + max(0, int(bytes_written))
    threshold = int(getattr(tensor, "_aetherscale_flush_threshold_bytes", _spill_flush_threshold_bytes()))
    if not force and pending < threshold:
        setattr(tensor, "_aetherscale_dirty_bytes", pending)
        return False
    try:
        from .progress import throw_if_interrupted
        throw_if_interrupted()
    except ImportError:
        pass
    try:
        mm.flush()
    finally:
        setattr(tensor, "_aetherscale_dirty_bytes", 0)
    # Do not empty the whole ComfyUI working set here. v0.8.7 used
    # SetProcessWorkingSetSize(-1,-1), which evicted unrelated model/input pages
    # and turned GPU stages into CPU/disk page-fault workloads. Flushing bounds
    # dirty writeback; Windows may reclaim clean mapped pages naturally.
    try:
        from .progress import throw_if_interrupted
        throw_if_interrupted()
    except ImportError:
        pass
    return True


@dataclass(slots=True)
class StorageInfo:
    backend: str
    dtype: str
    bytes: int
    path: str | None
    fallback_reason: str | None = None
    commit_available_bytes: int | None = None
    disk_free_bytes: int | None = None


def _pid_from_cache_name(path: Path) -> int | None:
    # Current cache names are: <prefix>_<timestamp_ms>_<pid>_<counter>.mmap
    try:
        parts = path.stem.rsplit("_", 3)
        if len(parts) != 4:
            return None
        return int(parts[-2])
    except (TypeError, ValueError):
        return None


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Windows can report access-related OSErrors for a live foreign process.
        return True
    return True


def _unlink_with_retry(path: str | Path, retries: int = 4, delay: float = 0.025) -> bool:
    p = Path(path)
    for attempt in range(max(1, int(retries))):
        try:
            p.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt + 1 < retries:
                time.sleep(delay)
        except OSError:
            if attempt + 1 < retries:
                time.sleep(delay)
    return not p.exists()


def _on_memmap_release(path: str, auto_delete: bool) -> None:
    with _CACHE_LOCK:
        _ACTIVE_PATHS.discard(path)
    if auto_delete:
        if _unlink_with_retry(path):
            with _CACHE_LOCK:
                _PENDING_DELETE.discard(path)
        else:
            # A Windows file mapping can remain delete-locked for a very short
            # period during finalization. Retry on the next cache operation.
            with _CACHE_LOCK:
                _PENDING_DELETE.add(path)


def _retry_pending_deletes() -> None:
    with _CACHE_LOCK:
        pending = list(_PENDING_DELETE)
    for path in pending:
        if _unlink_with_retry(path, retries=2):
            with _CACHE_LOCK:
                _PENDING_DELETE.discard(path)


def _cache_dirs() -> tuple[Path, ...]:
    dirs = [CACHE_DIR]
    if LEGACY_CACHE_DIR != CACHE_DIR:
        dirs.append(LEGACY_CACHE_DIR)
    # preserve order while deduplicating resolved paths
    seen: set[str] = set()
    out: list[Path] = []
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            out.append(directory)
    return tuple(out)


def _cleanup_orphaned_dir(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    with _CACHE_LOCK:
        active = set(_ACTIVE_PATHS)
    for p in directory.glob("*.mmap"):
        key = str(p)
        if key in active:
            continue
        owner_pid = _pid_from_cache_name(p)
        if owner_pid is not None and _pid_is_alive(owner_pid):
            continue
        # Files with a parseable dead PID are crash leftovers and are safe to
        # remove immediately. Unknown legacy names are left for age cleanup.
        if owner_pid is not None and _unlink_with_retry(p):
            removed += 1
    return removed


def cleanup_orphaned_cache() -> int:
    """Delete dead-process AetherScale mappings from current and legacy cache dirs."""
    _retry_pending_deletes()
    return sum(_cleanup_orphaned_dir(directory) for directory in _cache_dirs())


def cleanup_stale_cache(max_age_hours: float = 48.0) -> int:
    _retry_pending_deletes()
    cutoff = time.time() - max_age_hours * 3600.0
    removed = 0
    with _CACHE_LOCK:
        active = set(_ACTIVE_PATHS)
    for directory in _cache_dirs():
        directory.mkdir(parents=True, exist_ok=True)
        for p in directory.glob("*.mmap"):
            try:
                if str(p) in active:
                    continue
                owner_pid = _pid_from_cache_name(p)
                if owner_pid is not None and _pid_is_alive(owner_pid):
                    continue
                if p.stat().st_mtime < cutoff and _unlink_with_retry(p):
                    removed += 1
            except OSError:
                pass
    return removed


def startup_cache_cleanup() -> dict[str, object]:
    """Best-effort startup recovery for crash-leftover spill mappings.

    This runs while the custom node is imported so a previous 40+ GiB spill
    cannot silently keep ComfyUI's drive full and prevent the web UI/server
    from starting normally. Live-process mappings are never touched.
    """
    removed_orphans = cleanup_orphaned_cache()
    removed_stale = cleanup_stale_cache()
    return {
        "removed_orphans": int(removed_orphans),
        "removed_stale": int(removed_stale),
        "cache_dir": str(CACHE_DIR),
        "legacy_cache_dir": str(LEGACY_CACHE_DIR),
    }

def resolve_dtype(
    *,
    requested: str,
    shape: Sequence[int],
    input_dtype: torch.dtype,
    auto_fp16_threshold_mb: int = 768,
) -> torch.dtype:
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32

    # auto: long video outputs use FP16 to keep resident/commit pressure bounded.
    numel = 1
    for x in shape:
        numel *= int(x)
    float32_mb = numel * 4 / (1024 * 1024)
    if float32_mb >= float(auto_fp16_threshold_mb):
        return torch.float16

    # Keep conventional ComfyUI float32 when the output is small.
    if input_dtype in (torch.float16, torch.bfloat16):
        return input_dtype if input_dtype == torch.float16 else torch.float16
    return torch.float32


def estimate_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    numel = 1
    for x in shape:
        numel *= int(x)
    return int(numel * torch.empty((), dtype=dtype).element_size())



def _windows_commit_available_bytes() -> int | None:
    """Return Windows commit headroom (ullAvailPageFile), when available.

    Anonymous mmap on Windows is backed by the system paging/commit pool. A very
    large mapping can therefore fail with WinError 1450 even when physical RAM
    and disk free space look healthy. This probe lets auto storage spill to a
    file-backed mapping before Windows rejects the reservation.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return None
        return int(state.ullAvailPageFile)
    except Exception:
        return None


def _disk_free_bytes(path: Path) -> int | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return int(shutil.disk_usage(path).free)
    except Exception:
        return None


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def _create_file_backed_tensor(
    shape: tuple[int, ...],
    *,
    np_dtype,
    torch_dtype: torch.dtype,
    nbytes: int,
    prefix: str,
    auto_delete: bool,
    fallback_reason: str | None,
    commit_available: int | None,
) -> tuple[torch.Tensor, StorageInfo]:
    global _MMAP_COUNTER

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_free = _disk_free_bytes(CACHE_DIR)
    # Leave room for filesystem/OS activity. For very large outputs the fixed
    # 2 GiB floor is more useful than a tiny percentage-only margin.
    disk_margin = max(2 * 1024 ** 3, min(8 * 1024 ** 3, int(nbytes * 0.05)))
    if disk_free is not None and disk_free < nbytes + disk_margin:
        raise RuntimeError(
            "AetherScale cannot allocate the output safely: Windows commit is "
            f"insufficient for an anonymous mapping and disk spill also lacks space. "
            f"Output requires {_gib(nbytes)}; free spill disk space is {_gib(disk_free)} "
            f"in {CACHE_DIR}. Free disk space, reduce frame count/resolution, place MFG "
            "after spatial enhancement and use AetherScale MFG Video for direct encode, "
            "or set AETHERSCALE_CACHE_DIR to a drive with enough free space."
        )

    with _CACHE_LOCK:
        counter = _MMAP_COUNTER
        _MMAP_COUNTER += 1
    kind = "spill" if auto_delete else "mmap"
    stamp = f"{int(time.time()*1000)}_{os.getpid()}_{counter}"
    path = CACHE_DIR / f"{prefix}_{kind}_{stamp}.mmap"
    mm = np.memmap(path, mode="w+", dtype=np_dtype, shape=shape)
    path_s = str(path)
    with _CACHE_LOCK:
        _ACTIVE_PATHS.add(path_s)
    weakref.finalize(mm, _on_memmap_release, path_s, bool(auto_delete))
    t = torch.from_numpy(mm)
    # Keep an explicit handle so writers can periodically FlushViewOfFile via
    # numpy.memmap.flush(). This prevents giant dirty-page backlogs on Windows.
    setattr(t, "_aetherscale_memmap", mm)
    setattr(t, "_aetherscale_dirty_bytes", 0)
    setattr(t, "_aetherscale_flush_threshold_bytes", _spill_flush_threshold_bytes())
    backend = "disk_spill_mmap" if auto_delete else "mmap"
    return t, StorageInfo(
        backend,
        str(torch_dtype).replace("torch.", ""),
        nbytes,
        path_s,
        fallback_reason=fallback_reason,
        commit_available_bytes=commit_available,
        disk_free_bytes=disk_free,
    )



def make_cache_file_path(prefix: str, *, auto_delete: bool = True) -> Path:
    """Return a PID-tagged cache path without mapping/allocating it yet.

    Streaming producers (video decode, future stream processors) can write the
    file sequentially with ordinary file I/O, which avoids building a giant
    dirty writable mmap working set on Windows. Map it only after writing.
    """
    global _MMAP_COUNTER
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        counter = _MMAP_COUNTER
        _MMAP_COUNTER += 1
    kind = "spill" if auto_delete else "mmap"
    stamp = f"{int(time.time()*1000)}_{os.getpid()}_{counter}"
    return CACHE_DIR / f"{prefix}_{kind}_{stamp}.mmap"


def map_existing_file_tensor(
    path: str | Path,
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    auto_delete: bool = True,
    backend: str = "stream_file_mmap",
) -> tuple[torch.Tensor, StorageInfo]:
    """Map a fully-written tensor file without rewriting or zero-filling it."""
    p = Path(path).resolve()
    shape_t = tuple(int(x) for x in shape)
    np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported mmap dtype: {dtype}")
    nbytes = estimate_bytes(shape_t, dtype)
    if not p.is_file():
        raise FileNotFoundError(p)
    actual = p.stat().st_size
    if actual != nbytes:
        raise RuntimeError(
            f"AetherScale tensor file size mismatch for {p}: {actual} bytes, expected {nbytes}."
        )
    mm = np.memmap(p, mode="r+", dtype=np_dtype, shape=shape_t)
    path_s = str(p)
    with _CACHE_LOCK:
        _ACTIVE_PATHS.add(path_s)
    weakref.finalize(mm, _on_memmap_release, path_s, bool(auto_delete))
    t = torch.from_numpy(mm)
    setattr(t, "_aetherscale_memmap", mm)
    setattr(t, "_aetherscale_dirty_bytes", 0)
    setattr(t, "_aetherscale_flush_threshold_bytes", _spill_flush_threshold_bytes())
    return t, StorageInfo(
        str(backend),
        str(dtype).replace("torch.", ""),
        int(nbytes),
        path_s,
        fallback_reason=None,
        commit_available_bytes=_windows_commit_available_bytes(),
        disk_free_bytes=_disk_free_bytes(CACHE_DIR),
    )

def allocate_cpu_tensor(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    storage_mode: str = "auto",
    prefix: str = "output",
    mmap_threshold_mb: int = 768,
    clean_cache: bool = True,
) -> tuple[torch.Tensor, StorageInfo]:
    global _MMAP_COUNTER

    # Always remove old dead-process leftovers. With clean_cache=True, new
    # large outputs use anonymous mappings and therefore create no cache file.
    cleanup_orphaned_cache()
    cleanup_stale_cache()

    shape = tuple(int(x) for x in shape)
    nbytes = estimate_bytes(shape, dtype)
    threshold = int(mmap_threshold_mb) * 1024 * 1024

    use_mmap = storage_mode == "mmap" or (
        storage_mode == "auto" and nbytes >= threshold
    )
    if storage_mode == "ram":
        use_mmap = False

    if not use_mmap:
        t = torch.empty(shape, dtype=dtype, device="cpu")
        return t, StorageInfo("ram", str(dtype).replace("torch.", ""), nbytes, None)

    np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported mmap dtype: {dtype}")

    # clean_cache=True normally uses an anonymous/pagefile-backed mapping so no
    # cache file is created. On Windows, however, anonymous mmap consumes system
    # commit. Huge MFG -> SR pipelines can legitimately exceed the remaining
    # commit limit and fail with WinError 1450. In that case we automatically
    # spill to a file-backed mmap rather than failing the workflow.
    commit_available = _windows_commit_available_bytes()
    commit_margin = max(2 * 1024 ** 3, min(8 * 1024 ** 3, int(nbytes * 0.05)))
    force_spill = str(os.environ.get("AETHERSCALE_FORCE_DISK_SPILL", "")).strip().lower() in {
        "1", "true", "yes", "on"
    }

    if clean_cache:
        proactive_reason = None
        if force_spill:
            proactive_reason = "forced_by_AETHERSCALE_FORCE_DISK_SPILL"
        elif commit_available is not None and commit_available < nbytes + commit_margin:
            proactive_reason = (
                f"windows_commit_preflight: need {_gib(nbytes)} + margin; "
                f"available {_gib(commit_available)}"
            )

        if proactive_reason is None:
            try:
                backing = py_mmap.mmap(-1, nbytes, access=py_mmap.ACCESS_WRITE)
                arr = np.ndarray(shape=shape, dtype=np_dtype, buffer=backing)
                t = torch.from_numpy(arr)
                return t, StorageInfo(
                    "anonymous_mmap",
                    str(dtype).replace("torch.", ""),
                    nbytes,
                    None,
                    fallback_reason=None,
                    commit_available_bytes=commit_available,
                    disk_free_bytes=_disk_free_bytes(CACHE_DIR),
                )
            except (OSError, MemoryError) as exc:
                winerror = getattr(exc, "winerror", None)
                proactive_reason = (
                    f"anonymous_mmap_failed: {type(exc).__name__}"
                    + (f" WinError {winerror}" if winerror is not None else "")
                    + f": {exc}"
                )

        print(
            "[AetherScale] Large anonymous output mapping cannot be reserved; "
            f"spilling {_gib(nbytes)} to disk-backed mmap. Reason: {proactive_reason}"
        )
        return _create_file_backed_tensor(
            shape,
            np_dtype=np_dtype,
            torch_dtype=dtype,
            nbytes=nbytes,
            prefix=prefix,
            auto_delete=True,
            fallback_reason=proactive_reason,
            commit_available=commit_available,
        )

    # clean_cache=False is the explicit persistent-cache/debug mode.
    return _create_file_backed_tensor(
        shape,
        np_dtype=np_dtype,
        torch_dtype=dtype,
        nbytes=nbytes,
        prefix=prefix,
        auto_delete=False,
        fallback_reason=None,
        commit_available=commit_available,
    )
