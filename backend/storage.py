from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Sequence
import weakref

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".aetherscale_cache"
_CACHE_LOCK = threading.RLock()
_ACTIVE_PATHS: set[str] = set()
_PENDING_DELETE: set[str] = set()
_MMAP_COUNTER = 0


@dataclass(slots=True)
class StorageInfo:
    backend: str
    dtype: str
    bytes: int
    path: str | None


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


def cleanup_orphaned_cache() -> int:
    """Delete mmap files that are not owned by a live AetherScale process.

    This is intentionally PID-aware so two ComfyUI processes sharing the same
    custom-node folder do not delete each other's live mappings.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _retry_pending_deletes()
    removed = 0
    with _CACHE_LOCK:
        active = set(_ACTIVE_PATHS)
    for p in CACHE_DIR.glob("*.mmap"):
        key = str(p)
        if key in active:
            continue
        owner_pid = _pid_from_cache_name(p)
        if owner_pid is not None and _pid_is_alive(owner_pid):
            continue
        if _unlink_with_retry(p):
            removed += 1
    return removed


def cleanup_stale_cache(max_age_hours: float = 48.0) -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _retry_pending_deletes()
    cutoff = time.time() - max_age_hours * 3600.0
    removed = 0
    with _CACHE_LOCK:
        active = set(_ACTIVE_PATHS)
    for p in CACHE_DIR.glob("*.mmap"):
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

    # Always remove old dead-process leftovers. clean_cache additionally means
    # every newly-created mmap is delete-on-release rather than persistent.
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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        counter = _MMAP_COUNTER
        _MMAP_COUNTER += 1
    stamp = f"{int(time.time()*1000)}_{os.getpid()}_{counter}"
    path = CACHE_DIR / f"{prefix}_{stamp}.mmap"

    np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported mmap dtype: {dtype}")

    mm = np.memmap(path, mode="w+", dtype=np_dtype, shape=shape)
    path_s = str(path)
    with _CACHE_LOCK:
        _ACTIVE_PATHS.add(path_s)

    # torch.from_numpy keeps the NumPy owner alive for the lifetime of the
    # tensor storage (including downstream views). Finalizing the memmap is
    # therefore a safe point to remove the file. This fixes the previous
    # process-lifetime _MMAP_KEEPALIVE behaviour that leaked multi-GB files.
    weakref.finalize(mm, _on_memmap_release, path_s, bool(clean_cache))
    t = torch.from_numpy(mm)
    return t, StorageInfo("mmap", str(dtype).replace("torch.", ""), nbytes, path_s)
