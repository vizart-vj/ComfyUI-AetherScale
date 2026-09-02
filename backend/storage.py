from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".aetherscale_cache"
_MMAP_KEEPALIVE: list[np.memmap] = []


@dataclass(slots=True)
class StorageInfo:
    backend: str
    dtype: str
    bytes: int
    path: str | None


def cleanup_stale_cache(max_age_hours: float = 48.0) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age_hours * 3600.0
    for p in CACHE_DIR.glob("*.mmap"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


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
) -> tuple[torch.Tensor, StorageInfo]:
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
    stamp = f"{int(time.time()*1000)}_{os.getpid()}_{len(_MMAP_KEEPALIVE)}"
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
    _MMAP_KEEPALIVE.append(mm)
    t = torch.from_numpy(mm)
    return t, StorageInfo("mmap", str(dtype).replace("torch.", ""), nbytes, str(path))
