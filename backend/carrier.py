from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import threading
import time
import urllib.request
import zipfile

try:
    import winreg
except ImportError:
    winreg = None
from typing import Any, Optional

import numpy as np
import torch

from .storage import allocate_cpu_tensor, resolve_dtype

ROOT = Path(__file__).resolve().parents[1]
CARRIER_ROOT = ROOT / "runtime" / "carrier"
CARRIER_RUNTIME = CARRIER_ROOT / "runtime"
WORKER = CARRIER_RUNTIME / "nvngx.dll"
CARRIER_MANIFEST = CARRIER_ROOT / "carrier_manifest.json"

# Official GitHub release asset published by the upstream project author.
UPSTREAM_RELEASE = "v1.0"
UPSTREAM_URL = (
    "https://github.com/Merserk/dlss5-visual-enhancer/releases/download/"
    "v1.0/DLSS.5.Visual.Enhancer.v1.0.zip"
)
UPSTREAM_SHA256 = "5d57c2f2d2a1c247c0249e7a1024eabb5384ee9111820a4a478be6ce893b767d"

REQUIRED_RUNTIME = (
    "nvngx.dll",
    "dxgi.dll",
    "renodx-dlss5.addon64",
    "nvngx_dlss.dll",
    "nvngx_dlssnr.dll",
    "ReShade.ini",
)

VIDEO_MAGIC = 0x33563544
SETUP_MAGIC = 0x33505553
FRAME_MAGIC = 0x314D5246
OUT_MAGIC = 0x3154554F
VIDEO_HEADER_FORMAT = "<13I4f"
SETUP_RESPONSE_FORMAT = "<11I"
FRAME_HEADER_FORMAT = "<4Iq"
OUT_HEADER_FORMAT = "<5Iq"

UPSCALING_MODES = {
    "native_1x": (1.0, 5),
    "quality_1_5x": (1.5, 2),
    "balanced_1_724x": (1.724, 1),
    "performance_2x": (2.0, 0),
    "ultra_performance_3x": (3.0, 3),
}

_lock = threading.RLock()


class CarrierError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-AetherScale/0.5.4"})
    with urllib.request.urlopen(req, timeout=240) as resp, path.open("wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _runtime_ready() -> bool:
    return all((CARRIER_RUNTIME / name).is_file() for name in REQUIRED_RUNTIME)


def ensure_carrier(auto_bootstrap: bool = True, force: bool = False) -> dict[str, Any]:
    """Install only the runtime subset from Merserk's complete portable release."""
    with _lock:
        if _runtime_ready() and not force:
            return {
                "ready": True,
                "source": "Merserk/dlss5-visual-enhancer",
                "release": UPSTREAM_RELEASE,
                "runtime": str(CARRIER_RUNTIME),
                "worker": str(WORKER),
            }

        if not auto_bootstrap:
            raise CarrierError(
                "DLSS5 carrier runtime is missing. Enable auto_bootstrap so AetherScale "
                "can install the pinned upstream portable runtime automatically."
            )

        CARRIER_RUNTIME.mkdir(parents=True, exist_ok=True)
        archive = CARRIER_ROOT / ".visual-enhancer-v1.0.zip"
        print("[AetherScale] Installing DLSS5 carrier backend from Visual Enhancer v1.0...")
        try:
            _download(UPSTREAM_URL, archive)
            digest = _sha256_file(archive)
            if digest.lower() != UPSTREAM_SHA256.lower():
                raise CarrierError(
                    "Visual Enhancer release checksum mismatch: "
                    f"expected {UPSTREAM_SHA256}, got {digest}."
                )

            found: dict[str, str] = {}
            with zipfile.ZipFile(archive, "r") as zf:
                # Locate exact runtime filenames anywhere in the portable archive.
                members = zf.namelist()
                for required in REQUIRED_RUNTIME:
                    matches = [
                        m for m in members
                        if Path(m).name.lower() == required.lower()
                        and "/runtime/" in m.replace("\\", "/").lower()
                    ]
                    if not matches:
                        # ReShade.ini can occasionally be at a nearby portable root;
                        # all binary components must still come from runtime/.
                        matches = [
                            m for m in members
                            if Path(m).name.lower() == required.lower()
                        ]
                    if not matches:
                        raise CarrierError(
                            f"Visual Enhancer {UPSTREAM_RELEASE} archive does not contain {required}."
                        )
                    member = sorted(matches, key=len)[0]
                    data = zf.read(member)
                    dst = CARRIER_RUNTIME / required
                    dst.write_bytes(data)
                    found[required] = member

            if not _runtime_ready():
                raise CarrierError("Carrier extraction finished but runtime is incomplete.")

            manifest = {
                "source": "Merserk/dlss5-visual-enhancer",
                "release": UPSTREAM_RELEASE,
                "archive_url": UPSTREAM_URL,
                "archive_sha256": digest,
                "files": {
                    name: {
                        "archive_member": found[name],
                        "sha256": _sha256_file(CARRIER_RUNTIME / name),
                        "bytes": (CARRIER_RUNTIME / name).stat().st_size,
                    }
                    for name in REQUIRED_RUNTIME
                },
            }
            CARRIER_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print("[AetherScale] DLSS5 carrier backend ready.")
            return {
                "ready": True,
                "source": manifest["source"],
                "release": UPSTREAM_RELEASE,
                "runtime": str(CARRIER_RUNTIME),
                "worker": str(WORKER),
                "archive_sha256": digest,
            }
        finally:
            try:
                archive.unlink(missing_ok=True)
            except OSError:
                pass


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise EOFError(f"carrier worker stopped after {len(data)} of {size} bytes")
        data.extend(block)
    return bytes(data)


def _output_size(width: int, height: int, mode_name: str) -> tuple[int, int, float, int]:
    factor, perf_quality = UPSCALING_MODES[mode_name]
    def even(v: float) -> int:
        return max(2, int(math.floor(v / 2.0 + 0.5)) * 2)
    ow, oh = even(width * factor), even(height * factor)
    if max(ow, oh) > 7680 or min(ow, oh) > 4320:
        raise CarrierError(
            f"Requested {ow}x{oh} exceeds carrier 8K boundary for mode {mode_name}."
        )
    return ow, oh, factor, perf_quality


def _scene_cut_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


class _TemporalGuide:
    """OpenCV DIS current->previous motion matching the working Visual Enhancer contract."""

    def __init__(self, width: int, height: int, flow_width: int = 640):
        try:
            import cv2
        except Exception as exc:
            raise CarrierError(
                "Carrier internal motion requires OpenCV. Install opencv-python-headless "
                "or connect AetherScale Motion Analysis."
            ) from exc
        self.cv2 = cv2
        self.width = width
        self.height = height
        scale = min(1.0, flow_width / max(width, 1))
        self.fw = max(64, int(round(width * scale / 2.0) * 2))
        self.fh = max(64, int(round(height * scale / 2.0) * 2))
        self.previous = None
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.dis.setUseSpatialPropagation(True)
        self.dis.setFinestScale(1)

    def _gray(self, rgba8: np.ndarray) -> np.ndarray:
        gray = self.cv2.cvtColor(rgba8, self.cv2.COLOR_RGBA2GRAY)
        return self.cv2.resize(gray, (self.fw, self.fh), interpolation=self.cv2.INTER_AREA)

    def process(self, rgba8: np.ndarray, threshold: float = 0.24):
        cur = self._gray(rgba8)
        if self.previous is None:
            motion = np.zeros((self.height, self.width, 2), dtype=np.float16)
            reset = True
            score = 1.0
        else:
            score = _scene_cut_score(cur, self.previous)
            reset = score > threshold
            if reset:
                motion = np.zeros((self.height, self.width, 2), dtype=np.float16)
            else:
                # Important direction: CURRENT -> PREVIOUS.
                cur_to_prev = self.dis.calc(cur, self.previous, None)
                motion = self.cv2.resize(
                    cur_to_prev, (self.width, self.height),
                    interpolation=self.cv2.INTER_LINEAR
                )
                motion[..., 0] *= self.width / self.fw
                motion[..., 1] *= self.height / self.fh
                motion = np.ascontiguousarray(motion.astype(np.float16))
        self.previous = cur
        return motion, reset, score


def _motion_from_packet(packet: Any, frame_index: int, width: int, height: int):
    """Return current->previous float16 HxWx2 if packet carries compatible motion."""
    if packet is None:
        return None

    flow = getattr(packet, "flow", None)
    metadata = getattr(packet, "metadata", {}) or {}
    if not isinstance(flow, torch.Tensor) or flow.numel() == 0:
        return None

    direction = str(metadata.get("direction", ""))
    if direction not in ("current_to_previous", "cur_to_prev"):
        return None

    # frame_index=0 has no previous frame.
    if frame_index <= 0:
        return np.zeros((height, width, 2), dtype=np.float16)
    pair = frame_index - 1
    if pair >= int(flow.shape[0]):
        return None

    f = flow[pair].detach().to("cpu", dtype=torch.float32).numpy()
    fh, fw = f.shape[:2]
    if (fw, fh) != (width, height):
        try:
            import cv2
        except Exception:
            return None
        f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
        sx = float(metadata.get("scale_x_to_source", width / max(fw, 1)))
        sy = float(metadata.get("scale_y_to_source", height / max(fh, 1)))
        f[..., 0] *= sx
        f[..., 1] *= sy
    return np.ascontiguousarray(f.astype(np.float16))


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    """Enumerate physical NVIDIA GPUs independently of PyTorch visibility."""
    out: list[dict[str, Any]] = []
    if os.name != "nt":
        return out
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,pci.bus_id,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if cp.returncode != 0:
            return out
        for line in cp.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                out.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "pci_bus_id": parts[2],
                        "memory_mb": int(float(parts[3])),
                    }
                )
    except Exception:
        pass
    return out


def carrier_gpu_choices() -> list[str]:
    gpus = _nvidia_smi_gpus()
    if not gpus:
        return ["windows_high_performance"]
    # Highest-performance policy remains first/default. Physical choices are
    # provided for diagnostics/future native-worker selector support.
    choices = ["windows_high_performance"]
    choices += [
        f"gpu_{g['index']} | {g['name']} | {g['pci_bus_id']}"
        for g in gpus
    ]
    return choices


def _set_windows_gpu_preference(executable: Path, preference: str) -> dict[str, Any]:
    """Pin the carrier executable using the same per-app preference Windows Settings uses.

    Registry:
      HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences
      value-name = absolute executable path
      value-data = GpuPreference=2;  (High performance)

    This is process/app-scoped and does not disable another GPU system-wide.
    """
    info = {
        "requested": preference,
        "worker": str(executable.resolve()),
        "applied": False,
        "registry_value": None,
    }
    if os.name != "nt" or winreg is None:
        info["reason"] = "not_windows"
        return info

    # The current closed carrier worker has no public --gpu selector. Therefore
    # an explicit gpu_N request maps to Windows High Performance as the only
    # reliable external D3D12 routing control. We still record the requested
    # physical GPU so logs make this limitation explicit.
    value = "GpuPreference=2;"
    key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            key_path,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
        ) as key:
            winreg.SetValueEx(
                key,
                str(executable.resolve()),
                0,
                winreg.REG_SZ,
                value,
            )
        info["applied"] = True
        info["registry_value"] = value
        info["available_nvidia_gpus"] = _nvidia_smi_gpus()
        return info
    except Exception as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return info


def _high_performance_gpu() -> dict[str, Any] | None:
    """Best expected Windows high-performance NVIDIA GPU.

    Since the carrier worker itself chooses through D3D12, this is diagnostic:
    among discrete NVIDIA cards we rank by known architecture naming, then VRAM.
    """
    gpus = _nvidia_smi_gpus()
    if not gpus:
        return None

    def score(g: dict[str, Any]) -> tuple[int, int]:
        name = str(g["name"]).upper()
        generation = 0
        m = re.search(r"RTX\s+(\d{2})", name)
        if m:
            generation = int(m.group(1))
        return generation, int(g.get("memory_mb", 0))

    return max(gpus, key=score)


def process_carrier(
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
    upscale_mode: str,
    warmup_frames: int,
    scene_cut_threshold: float,
    motion_source: str,
    auto_bootstrap: bool,
    output_precision: str,
    output_storage: str,
    clean_cache: bool = True,
    carrier_gpu: str = "windows_high_performance",
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise CarrierError(f"Expected IMAGE [B,H,W,C], got {tuple(images.shape)}")
    if images.shape[-1] < 3:
        raise CarrierError("Carrier requires RGB/RGBA IMAGE input.")

    ensure = ensure_carrier(auto_bootstrap=auto_bootstrap)
    batch, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    ow, oh, factor, perf_quality = _output_size(w, h, upscale_mode)

    style_map = {
        "auto": 0,
        "default": 0,
        "natural": 1,
        "cinematic": 2,
        "material_detail": 0,
        "3": 3, "4": 4, "5": 5, "6": 6,
    }
    style_i = style_map.get(str(style), 0)

    out_shape = (batch, oh, ow, 4 if images.shape[-1] > 3 else 3)
    out_dtype = resolve_dtype(
        requested=output_precision,
        shape=out_shape,
        input_dtype=images.dtype,
    )
    out_cpu, storage = allocate_cpu_tensor(
        out_shape,
        dtype=out_dtype,
        storage_mode=output_storage,
        prefix="carrier_dlss5",
        clean_cache=bool(clean_cache),
    )

    # Carrier is D3D12/DXGI, not CUDA. Pin the worker through Windows'
    # per-application GPU preference before process creation.
    gpu_routing = _set_windows_gpu_preference(WORKER, carrier_gpu)
    expected_gpu = _high_performance_gpu()
    if gpu_routing.get("applied"):
        print(
            "[AetherScale] Carrier GPU routing: Windows High Performance -> "
            + (expected_gpu["name"] if expected_gpu else "best available adapter")
        )

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [str(WORKER), "--video"],
        cwd=str(CARRIER_RUNTIME),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )
    assert proc.stdin and proc.stdout and proc.stderr

    stderr_lines: list[str] = []
    def drain():
        for raw in iter(proc.stderr.readline, b""):
            stderr_lines.append(raw.decode("utf-8", "replace").rstrip())
    t = threading.Thread(target=drain, daemon=True)
    t.start()

    header = struct.pack(
        VIDEO_HEADER_FORMAT,
        VIDEO_MAGIC,
        w, h, ow, oh,
        int(warmup_frames),
        batch,
        int(perf_quality),
        0,                 # profile
        int(preset),
        int(style_i),
        1 if auto_mask else 0,
        0,                 # ui correction
        float(intensity),
        float(tone),
        float(structure),
        float(skin),
    )

    internal_guide = None
    if motion_source in ("auto", "internal_dis"):
        try:
            internal_guide = _TemporalGuide(w, h, flow_width=640)
        except CarrierError:
            if motion_source == "internal_dis":
                proc.terminate()
                raise

    scene_scores: list[float] = []
    reset_frames: list[int] = []
    render_w = render_h = min_w = min_h = max_w = max_h = 0

    try:
        proc.stdin.write(header)
        proc.stdin.flush()
        try:
            setup = _read_exact(proc.stdout, struct.calcsize(SETUP_RESPONSE_FORMAT))
        except EOFError as exc:
            proc.wait(timeout=10)
            t.join(timeout=2)
            raise CarrierError(
                "Carrier worker failed during DLSS setup:\n"
                + ("\n".join(stderr_lines[-80:]) or "no worker diagnostics")
            ) from exc

        (
            setup_magic, setup_ok, setup_result,
            render_w, render_h, nego_ow, nego_oh,
            min_w, min_h, max_w, max_h,
        ) = struct.unpack(SETUP_RESPONSE_FORMAT, setup)

        if setup_magic != SETUP_MAGIC:
            raise CarrierError("Carrier worker protocol mismatch.")
        if not setup_ok:
            raise CarrierError(
                f"DLSS carrier setup rejected mode {upscale_mode}: "
                f"NGX 0x{setup_result:08X}\n" + "\n".join(stderr_lines[-80:])
            )
        if (nego_ow, nego_oh) != (ow, oh):
            raise CarrierError(
                f"Carrier negotiated {nego_ow}x{nego_oh}, expected {ow}x{oh}."
            )

        for i in range(batch):
            rgb = images[i, ..., :3].detach().to("cpu", dtype=torch.float32)
            rgb8 = (rgb.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = rgb8
            rgba[..., 3] = 255

            packet_motion = None
            if motion_source in ("auto", "connected_motion"):
                packet_motion = _motion_from_packet(motion, i, w, h)

            if packet_motion is not None:
                mv = packet_motion
                cuts = getattr(motion, "scene_cuts", None)
                reset = i == 0
                if i > 0 and isinstance(cuts, torch.Tensor) and (i-1) < cuts.numel():
                    reset = bool(cuts[i-1].item())
                score = 1.0 if reset else 0.0
            elif internal_guide is not None:
                mv, reset, score = internal_guide.process(
                    rgba, threshold=float(scene_cut_threshold)
                )
            else:
                mv = np.zeros((h, w, 2), dtype=np.float16)
                reset = i == 0
                score = 1.0 if reset else 0.0

            if reset:
                reset_frames.append(i)
            scene_scores.append(float(score))

            frame_header = struct.pack(
                FRAME_HEADER_FORMAT,
                FRAME_MAGIC, i, int(reset), 0, i
            )
            proc.stdin.write(frame_header)
            proc.stdin.write(np.ascontiguousarray(rgba).tobytes())
            proc.stdin.write(np.ascontiguousarray(mv, dtype=np.float16).tobytes())
            proc.stdin.flush()

            result_header = _read_exact(proc.stdout, struct.calcsize(OUT_HEADER_FORMAT))
            magic, out_index, ok, byte_count, ngx_result, out_pts = struct.unpack(
                OUT_HEADER_FORMAT, result_header
            )
            expected = ow * oh * 4
            if magic != OUT_MAGIC or out_index != i or byte_count != expected:
                raise CarrierError(f"Invalid carrier response at frame {i}.")
            if not ok or ngx_result != 1:
                raise CarrierError(
                    f"Carrier feature-18 evaluation failed at frame {i}: "
                    f"NGX 0x{ngx_result:08X}\n" + "\n".join(stderr_lines[-80:])
                )

            arr = np.frombuffer(
                _read_exact(proc.stdout, byte_count), dtype=np.uint8
            ).reshape(oh, ow, 4)
            out_cpu[i, ..., :3].copy_(
                torch.from_numpy(arr[..., :3].copy()).to(dtype=out_dtype).div_(255.0)
            )
            if out_shape[-1] == 4:
                # Preserve input alpha, scaled independently.
                try:
                    import torch.nn.functional as F
                    alpha = images[i:i+1, ..., 3:4].detach().to("cpu", dtype=torch.float32)
                    alpha = F.interpolate(
                        alpha.permute(0,3,1,2), size=(oh,ow),
                        mode="bilinear", align_corners=False
                    ).permute(0,2,3,1)[0]
                    out_cpu[i, ..., 3:4].copy_(alpha.to(dtype=out_dtype))
                except Exception:
                    out_cpu[i, ..., 3].fill_(1)

        proc.stdin.close()
        code = proc.wait(timeout=60)
        t.join(timeout=2)
        if code:
            raise CarrierError(
                f"Carrier worker exited with code {code}:\n" + "\n".join(stderr_lines[-80:])
            )
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        raise

    return out_cpu, {
        "backend": "carrier_dlss5",
        "carrier_source": ensure["source"],
        "carrier_release": ensure["release"],
        "frames": batch,
        "input_resolution": [w, h],
        "render_resolution": [render_w, render_h],
        "output_resolution": [ow, oh],
        "upscale_mode": upscale_mode,
        "factor": factor,
        "perf_quality": perf_quality,
        "preset": int(preset),
        "style": style,
        "motion_source": motion_source,
        "scene_cut_threshold": float(scene_cut_threshold),
        "reset_frames": reset_frames,
        "scene_scores": [round(x, 6) for x in scene_scores],
        "warmup_frames": int(warmup_frames),
        "worker_protocol": "visual-enhancer-v3",
        "output_precision": storage.dtype,
        "output_storage_backend": storage.backend,
        "output_storage_path": storage.path,
        "clean_cache": bool(clean_cache),
        "output_gib": round(storage.bytes / 1024**3, 3),
        "worker_log_tail": stderr_lines[-20:],
        "legacy_direct_used": False,
        "carrier_gpu_request": carrier_gpu,
        "carrier_gpu_routing": gpu_routing,
        "expected_high_performance_gpu": expected_gpu,
        "nvidia_gpus": _nvidia_smi_gpus(),
    }
