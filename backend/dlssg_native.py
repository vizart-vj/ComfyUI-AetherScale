from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .download import DownloadFailure, download_file
from .neural import MotionPacket
from .storage import allocate_cpu_tensor, resolve_dtype, estimate_bytes, sync_file_backed_tensor
from .progress import ConsoleProgress

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime" / "dlssg_native"
WORKER = RUNTIME_DIR / "dlssg-worker.exe"
DLSSG_DLL = RUNTIME_DIR / "nvngx_dlssg.dll"
MANIFEST = RUNTIME_DIR / "manifest.json"

# Pinned public reference implementation revision. Git LFS media URLs resolve to
# the actual binary objects rather than the small pointer files in source ZIPs.
UPSTREAM_REPO = "Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation"
UPSTREAM_COMMIT = "2c5b661fb94a236321414300e6269441acb2d13d"
WORKER_SHA256 = "8a747f9ed613842d5b8b34a811ad43bc1a9466540e2e5a0c8ef4005f0db9e384"
WORKER_SIZE = 66560
DLL_SHA256 = "135eaf0733c1e37381a8c28abcf7a862404a54132b81787c04e35d09efc5e36f"
DLL_SIZE = 7519856
WORKER_MEDIA_URL = (
    "https://media.githubusercontent.com/media/"
    f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/bin/runtime/dlssg/dlssg-worker.exe"
)
WORKER_RAW_URL = (
    f"https://github.com/{UPSTREAM_REPO}/raw/{UPSTREAM_COMMIT}/"
    "bin/runtime/dlssg/dlssg-worker.exe"
)
DLL_MEDIA_URL = (
    "https://media.githubusercontent.com/media/"
    f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/bin/runtime/dlssg/nvngx_dlssg.dll"
)
DLL_RAW_URL = (
    f"https://github.com/{UPSTREAM_REPO}/raw/{UPSTREAM_COMMIT}/"
    "bin/runtime/dlssg/nvngx_dlssg.dll"
)
LICENSE_URL = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/"
    "bin/runtime/dlssg/LICENSE-NVIDIA-DLSS.txt"
)

SETUP_MAGIC = 0x31534746
SETUP_OUT_MAGIC = 0x31524746
FRAME_MAGIC = 0x31464746
FRAME_OUT_MAGIC = 0x314F4746

_RUNTIME_LOCK = threading.RLock()
_CAPABILITY_CACHE: dict[str, Any] | None = None


class NativeDLSSGError(RuntimeError):
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


def _valid_file(path: Path, sha256: str, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size and _sha256_file(path) == sha256
    except OSError:
        return False


def _copy_if_valid(source: Path, target: Path, sha256: str, size: int) -> bool:
    if not _valid_file(source, sha256, size):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def _local_candidates(filename: str) -> list[Path]:
    out: list[Path] = []
    env_dir = os.environ.get("AETHERSCALE_DLSSG_DIR", "").strip()
    if env_dir:
        out.append(Path(env_dir).expanduser() / filename)
    userprofile = os.environ.get("USERPROFILE", "").strip()
    if userprofile:
        out.append(Path(userprofile) / "Downloads" / filename)
    try:
        out.append(Path.home() / "Downloads" / filename)
    except Exception:
        pass
    return out


def _obtain_binary(
    target: Path,
    *,
    sha256: str,
    size: int,
    primary_url: str,
    alternate_url: str,
) -> dict[str, Any]:
    if _valid_file(target, sha256, size):
        return {"source": "existing", "path": str(target), "sha256": sha256}
    for candidate in _local_candidates(target.name):
        if _copy_if_valid(candidate, target, sha256, size):
            return {"source": f"local:{candidate}", "path": str(target), "sha256": sha256}

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = download_file(
            primary_url,
            target,
            user_agent="ComfyUI-AetherScale/0.9.2",
            timeout=240,
            extra_urls=(alternate_url,),
        )
    except DownloadFailure as exc:
        raise NativeDLSSGError(
            f"Unable to download native DLSSG component {target.name}: {exc}"
        ) from exc
    digest = _sha256_file(target)
    actual_size = target.stat().st_size
    if digest != sha256 or actual_size != size:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise NativeDLSSGError(
            f"Native DLSSG component verification failed for {target.name}: "
            f"expected size={size} sha256={sha256}, got size={actual_size} sha256={digest}."
        )
    return {
        "source": f"download:{meta.get('url', primary_url)}",
        "transport": meta.get("transport", "unknown"),
        "path": str(target),
        "sha256": digest,
    }


def _set_windows_high_performance(executable: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"applied": False, "executable": str(executable)}
    if os.name != "nt":
        result["reason"] = "not_windows"
        return result
    try:
        import winreg

        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(
                key,
                str(executable.resolve()),
                0,
                winreg.REG_SZ,
                "GpuPreference=2;",
            )
        result["applied"] = True
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def ensure_native_runtime(auto_bootstrap: bool = True) -> dict[str, Any]:
    with _RUNTIME_LOCK:
        if _valid_file(WORKER, WORKER_SHA256, WORKER_SIZE) and _valid_file(DLSSG_DLL, DLL_SHA256, DLL_SIZE):
            return {
                "ready": True,
                "source": UPSTREAM_REPO,
                "commit": UPSTREAM_COMMIT,
                "runtime_dir": str(RUNTIME_DIR),
                "worker": str(WORKER),
                "dll": str(DLSSG_DLL),
            }
        if not auto_bootstrap:
            raise NativeDLSSGError(
                "Native DLSSG runtime is missing. Enable native_auto_bootstrap or place "
                "the pinned dlssg-worker.exe and nvngx_dlssg.dll in runtime/dlssg_native."
            )
        print("[AetherScale] Installing native DLSS Frame Generation backend...")
        worker_meta = _obtain_binary(
            WORKER,
            sha256=WORKER_SHA256,
            size=WORKER_SIZE,
            primary_url=WORKER_MEDIA_URL,
            alternate_url=WORKER_RAW_URL,
        )
        dll_meta = _obtain_binary(
            DLSSG_DLL,
            sha256=DLL_SHA256,
            size=DLL_SIZE,
            primary_url=DLL_MEDIA_URL,
            alternate_url=DLL_RAW_URL,
        )
        # License download is best-effort because it is not executable/runtime data.
        license_path = RUNTIME_DIR / "LICENSE-NVIDIA-DLSS.txt"
        if not license_path.is_file():
            try:
                download_file(
                    LICENSE_URL,
                    license_path,
                    user_agent="ComfyUI-AetherScale/0.9.2",
                    timeout=60,
                )
            except Exception:
                pass
        manifest = {
            "source": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "worker": worker_meta,
            "dll": dll_meta,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "ready": True,
            "source": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "runtime_dir": str(RUNTIME_DIR),
            "worker": str(WORKER),
            "dll": str(DLSSG_DLL),
            "bootstrap": manifest,
        }


def _run_probe(worker_path: Path) -> dict[str, Any]:
    preference = _set_windows_high_performance(worker_path)
    proc = subprocess.run(
        [str(worker_path), "--probe"],
        cwd=str(worker_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise NativeDLSSGError(
            f"Native DLSSG capability probe exited with {proc.returncode}: {detail[-4000:]}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise NativeDLSSGError("Native DLSSG capability probe returned no JSON result.")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise NativeDLSSGError(
            "Native DLSSG capability probe returned invalid JSON: " + lines[-1][-2000:]
        ) from exc
    data["windows_gpu_preference"] = preference
    return data


def probe_native_capabilities(auto_bootstrap: bool = True, *, force: bool = False) -> dict[str, Any]:
    global _CAPABILITY_CACHE
    with _RUNTIME_LOCK:
        if _CAPABILITY_CACHE is not None and not force:
            return dict(_CAPABILITY_CACHE)
        runtime = ensure_native_runtime(auto_bootstrap=auto_bootstrap)
        if os.name != "nt":
            data = {
                "available": False,
                "detail": "Native DLSSG worker is Windows-only.",
                "multi_frame_count_max": 0,
                "native_multiplier": 1,
                "runtime": runtime,
            }
            _CAPABILITY_CACHE = data
            return dict(data)
        try:
            probed = _run_probe(WORKER)
            max_generated = max(0, int(probed.get("multi_frame_count_max", 0)))
            available = bool(probed.get("available", False)) and max_generated >= 1
            data = {
                **probed,
                "available": available,
                "multi_frame_count_max": max_generated,
                "native_multiplier": max_generated + 1 if available else 1,
                "runtime": runtime,
            }
        except Exception as exc:
            data = {
                "available": False,
                "detail": str(exc),
                "multi_frame_count_max": 0,
                "native_multiplier": 1,
                "runtime": runtime,
            }
        _CAPABILITY_CACHE = data
        return dict(data)


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise NativeDLSSGError("Native DLSSG worker closed its output unexpectedly.")
        data.extend(block)
    return bytes(data)


class DirectDLSSGSession:
    def __init__(
        self,
        width: int,
        height: int,
        frame_count: int,
        generated_count: int,
        *,
        worker_path: Path | None = None,
        runtime_dir: Path | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * 4
        self.generated_count = int(generated_count)
        self.worker_path = Path(worker_path or WORKER)
        self.runtime_dir = Path(runtime_dir or RUNTIME_DIR)
        self.logs: collections.deque[str] = collections.deque(maxlen=400)
        if not self.worker_path.is_file():
            raise NativeDLSSGError(f"Native DLSSG worker is missing: {self.worker_path}")
        _set_windows_high_performance(self.worker_path)
        self.process = subprocess.Popen(
            [str(self.worker_path), "--serve"],
            cwd=str(self.runtime_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._log_thread = threading.Thread(target=self._read_logs, daemon=True)
        self._log_thread.start()
        self.process.stdin.write(
            struct.pack(
                "<5I",
                SETUP_MAGIC,
                self.width,
                self.height,
                max(1, int(frame_count)),
                self.generated_count,
            )
        )
        self.process.stdin.flush()
        magic, status, maximum, _reserved = struct.unpack(
            "<4I", _read_exact(self.process.stdout, struct.calcsize("<4I"))
        )
        if magic != SETUP_OUT_MAGIC or status:
            detail = self.log_text()
            self.close()
            raise NativeDLSSGError(
                f"Native DLSSG session creation failed (status={status}, magic=0x{magic:08x}); "
                f"runtime maximum is {maximum + 1}x.\n{detail}"
            )
        if self.generated_count > int(maximum):
            self.close()
            raise NativeDLSSGError(
                f"Native DLSSG worker rejected {self.generated_count} generated frame(s); "
                f"MultiFrameCountMax is {maximum}."
            )
        self.maximum_generated = int(maximum)
        self._next_index = 0
        self.closed = False

    def _read_logs(self) -> None:
        if self.process.stderr is None:
            return
        for raw in iter(self.process.stderr.readline, b""):
            self.logs.append(raw.decode("utf-8", "replace").rstrip())

    def log_text(self) -> str:
        return "\n".join(self.logs)

    def process_frame(
        self,
        rgba: np.ndarray,
        motion: np.ndarray,
        timestamp: Fraction,
        *,
        reset: bool,
    ) -> list[np.ndarray]:
        if self.closed:
            raise NativeDLSSGError("Native DLSSG session is closed.")
        color = np.ascontiguousarray(rgba, dtype=np.uint8)
        vectors = np.ascontiguousarray(motion, dtype=np.float16)
        if color.shape != (self.height, self.width, 4):
            raise ValueError(f"DLSSG color frame has unexpected shape {color.shape}.")
        if vectors.shape != (self.height, self.width, 2):
            raise ValueError(f"DLSSG motion field has unexpected shape {vectors.shape}.")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(
            struct.pack(
                "<4I2q",
                FRAME_MAGIC,
                self._next_index,
                int(bool(reset)),
                0,
                int(timestamp.numerator),
                int(timestamp.denominator),
            )
        )
        self.process.stdin.write(memoryview(color).cast("B"))
        self.process.stdin.write(memoryview(vectors).cast("B"))
        self.process.stdin.flush()
        frame_index = self._next_index
        self._next_index += 1
        magic, status, generated, disabled = struct.unpack(
            "<4I", _read_exact(self.process.stdout, struct.calcsize("<4I"))
        )
        if magic != FRAME_OUT_MAGIC or status:
            raise NativeDLSSGError(
                f"Native DLSSG evaluation failed at input frame {frame_index} "
                f"(status={status}, magic=0x{magic:08x}).\n{self.log_text()}"
            )
        if disabled:
            return []
        return [
            np.frombuffer(_read_exact(self.process.stdout, self.frame_bytes), np.uint8)
            .reshape(self.height, self.width, 4)
            .copy()
            for _ in range(int(generated))
        ]

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        self.closed = True
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        try:
            self._log_thread.join(timeout=1)
        except Exception:
            pass

    def __enter__(self) -> "DirectDLSSGSession":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


@dataclass(slots=True)
class Guide:
    motion: np.ndarray
    reset: bool
    scene_score: float
    duplicate: bool
    confidence: float


class DISGuideGenerator:
    """Stable current->previous optical-flow guide matching the native reference path."""

    def __init__(self, width: int, height: int, flow_width: int = 640, scene_cut_threshold: float = 0.24) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise NativeDLSSGError(
                "Native DLSSG internal guide generation requires OpenCV (cv2), which is normally bundled with ComfyUI."
            ) from exc
        self.cv2 = cv2
        self.width = int(width)
        self.height = int(height)
        scale = min(1.0, int(flow_width) / max(1, self.width))
        self.flow_width = max(64, int(round(self.width * scale / 2) * 2))
        self.flow_height = max(64, int(round(self.height * scale / 2) * 2))
        self.previous: np.ndarray | None = None
        self.zero = np.zeros((self.height, self.width, 2), dtype=np.float16)
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.flow.setUseSpatialPropagation(True)
        self.flow.setFinestScale(1)

    def _gray(self, rgba: np.ndarray) -> np.ndarray:
        gray = self.cv2.cvtColor(rgba, self.cv2.COLOR_RGBA2GRAY)
        return self.cv2.resize(
            gray,
            (self.flow_width, self.flow_height),
            interpolation=self.cv2.INTER_AREA,
        )

    def process(self, rgba: np.ndarray, *, force_reset: bool = False, source_index: int | None = None) -> Guide:
        del source_index
        current = self._gray(rgba)
        if self.previous is None:
            guide = Guide(self.zero, True, 1.0, False, 0.0)
        else:
            difference = self.cv2.absdiff(current, self.previous)
            score = float(np.mean(difference)) / 255.0
            duplicate = score < 0.0005
            reset = bool(force_reset or score > self.scene_cut_threshold)
            if reset or duplicate:
                vectors = self.zero
                confidence = 1.0 if duplicate else 0.0
            else:
                calculated = self.flow.calc(current, self.previous, None)
                calculated = self.cv2.resize(
                    calculated,
                    (self.width, self.height),
                    interpolation=self.cv2.INTER_LINEAR,
                )
                calculated[..., 0] *= self.width / self.flow_width
                calculated[..., 1] *= self.height / self.flow_height
                finite = np.isfinite(calculated).all(axis=2)
                confidence = float(np.mean(finite))
                calculated[~finite] = 0
                reset = confidence < 0.98
                vectors = self.zero if reset else np.ascontiguousarray(calculated.astype(np.float16))
            guide = Guide(vectors, reset, score, duplicate, confidence)
        self.previous = current
        return guide


class PacketGuideGenerator:
    def __init__(self, packet: MotionPacket, width: int, height: int) -> None:
        self.packet = packet
        self.width = int(width)
        self.height = int(height)
        self.zero = np.zeros((self.height, self.width, 2), dtype=np.float16)
        self.first = True

    def _flow_for_pair(self, pair_index: int) -> np.ndarray:
        if pair_index < 0 or pair_index >= int(self.packet.flow.shape[0]):
            return self.zero
        flow_hwc = self.packet.flow[pair_index].to("cpu", dtype=torch.float32)
        src_h, src_w = int(flow_hwc.shape[0]), int(flow_hwc.shape[1])
        x = flow_hwc.permute(2, 0, 1).unsqueeze(0)
        if (src_h, src_w) != (self.height, self.width):
            x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)
            x[:, 0] *= float(self.width) / float(max(1, src_w))
            x[:, 1] *= float(self.height) / float(max(1, src_h))
        arr = x[0].permute(1, 2, 0).contiguous().numpy().astype(np.float16, copy=False)
        return np.ascontiguousarray(arr)

    def process(self, rgba: np.ndarray, *, force_reset: bool = False, source_index: int | None = None) -> Guide:
        del rgba
        if self.first:
            self.first = False
            return Guide(self.zero, True, 1.0, False, 0.0)
        if source_index is None:
            return Guide(self.zero, True, 1.0, False, 0.0)
        pair_index = int(source_index) - 1
        cut = bool(force_reset)
        if self.packet.scene_cuts.numel() and 0 <= pair_index < int(self.packet.scene_cuts.numel()):
            cut = cut or bool(self.packet.scene_cuts[pair_index].item())
        if cut:
            return Guide(self.zero, True, 1.0, False, 0.0)
        confidence = 1.0
        if self.packet.confidence.numel() and 0 <= pair_index < int(self.packet.confidence.shape[0]):
            confidence = float(self.packet.confidence[pair_index].float().mean().item())
        return Guide(self._flow_for_pair(pair_index), False, 0.0, False, confidence)


class ZeroGuideGenerator:
    def __init__(self, width: int, height: int) -> None:
        self.zero = np.zeros((int(height), int(width), 2), dtype=np.float16)
        self.first = True

    def process(self, rgba: np.ndarray, *, force_reset: bool = False, source_index: int | None = None) -> Guide:
        del rgba, source_index
        reset = self.first or bool(force_reset)
        self.first = False
        return Guide(self.zero, reset, 0.0, False, 1.0)


@dataclass(slots=True)
class TimedFrame:
    rgba: np.ndarray
    timestamp: Fraction
    segment: int
    provenance: str
    source_index: int | None = None


class NativeStage:
    def __init__(
        self,
        session: DirectDLSSGSession,
        width: int,
        height: int,
        generated_count: int,
        guide_generator: Any,
        *,
        detect_source_cuts: bool,
    ) -> None:
        self.session = session
        self.guide = guide_generator
        self.generated_count = int(generated_count)
        self.previous: TimedFrame | None = None
        self.detect_source_cuts = bool(detect_source_cuts)
        self.scene_cuts = 0
        self.duplicates = 0
        self.resets = 0
        self.generated = 0

    def push(self, frame: TimedFrame) -> list[TimedFrame]:
        force_reset = self.previous is not None and frame.segment != self.previous.segment
        guide = self.guide.process(
            frame.rgba,
            force_reset=force_reset,
            source_index=frame.source_index,
        )
        if self.previous is None:
            self.session.process_frame(frame.rgba, guide.motion, frame.timestamp, reset=True)
            self.previous = frame
            self.resets += 1
            return [frame]
        if self.detect_source_cuts and guide.reset and not force_reset:
            frame.segment = self.previous.segment + 1
            force_reset = True
            if not guide.duplicate:
                self.scene_cuts += 1
        if guide.duplicate:
            self.duplicates += 1
        reset = bool(force_reset or guide.reset)
        if reset:
            self.resets += 1
        generated = self.session.process_frame(
            frame.rgba,
            guide.motion,
            frame.timestamp,
            reset=reset,
        )
        output: list[TimedFrame] = []
        if not reset:
            interval = frame.timestamp - self.previous.timestamp
            count = len(generated)
            for index, rgba in enumerate(generated, start=1):
                # Use actual count returned by runtime; normal path equals requested generated_count.
                denominator = count + 1
                output.append(
                    TimedFrame(
                        rgba,
                        self.previous.timestamp + interval * Fraction(index, denominator),
                        frame.segment,
                        "DLSSG",
                        None,
                    )
                )
                self.generated += 1
        output.append(frame)
        self.previous = frame
        return output


class NearestTimelineWriter:
    def __init__(
        self,
        target_rate: Fraction,
        output_count: int,
        emit: Callable[[np.ndarray, int, Fraction, TimedFrame], None],
    ) -> None:
        self.target_rate = target_rate
        self.output_count = int(output_count)
        self.emit = emit
        self.next_index = 0
        self.previous: TimedFrame | None = None
        self.tie_late = False
        self.generated_selected = 0
        self.source_selected = 0
        self.max_error = Fraction(0)

    def _write(self, frame: TimedFrame, ideal: Fraction) -> None:
        self.emit(frame.rgba, self.next_index, ideal, frame)
        error = abs(frame.timestamp - ideal)
        self.max_error = max(self.max_error, error)
        if frame.provenance == "DLSSG":
            self.generated_selected += 1
        else:
            self.source_selected += 1
        self.next_index += 1

    def push(self, current: TimedFrame) -> None:
        if self.previous is None:
            self.previous = current
            return
        midpoint = (self.previous.timestamp + current.timestamp) / 2
        while self.next_index < self.output_count:
            ideal = Fraction(self.next_index, 1) / self.target_rate
            if ideal < midpoint:
                self._write(self.previous, ideal)
            elif ideal == midpoint:
                selected = current if self.tie_late else self.previous
                self.tie_late = not self.tie_late
                self._write(selected, ideal)
            else:
                break
        self.previous = current

    def finish(self) -> None:
        if self.previous is None:
            raise NativeDLSSGError("Native MFG received no frames.")
        while self.next_index < self.output_count:
            ideal = Fraction(self.next_index, 1) / self.target_rate
            endpoint = TimedFrame(
                self.previous.rgba,
                ideal,
                self.previous.segment,
                "Source",
                self.previous.source_index,
            )
            self._write(endpoint, ideal)


def _frame_to_rgba8(frame_hwc: torch.Tensor) -> np.ndarray:
    x = frame_hwc.detach()
    if x.device.type != "cpu":
        x = x.to("cpu", non_blocking=False)
    if x.is_floating_point():
        x = (x.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    elif x.dtype != torch.uint8:
        x = x.to(torch.uint8)
    x = x.contiguous()
    arr = x.numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC frame, got {arr.shape}")
    channels = int(arr.shape[2])
    if channels == 4:
        return np.ascontiguousarray(arr)
    if channels == 3:
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate((arr, alpha), axis=2))
    if channels == 1:
        rgb = np.repeat(arr, 3, axis=2)
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
    raise ValueError(f"Native DLSSG supports 1, 3, or 4 channel IMAGE inputs, got {channels}.")


def _make_guide(
    source: str,
    width: int,
    height: int,
    packet: MotionPacket | None,
    scene_cut_threshold: float,
) -> Any:
    if source == "connected_motion" and packet is not None and packet.flow.numel() > 0:
        return PacketGuideGenerator(packet, width, height)
    if source == "zero_motion":
        return ZeroGuideGenerator(width, height)
    return DISGuideGenerator(width, height, flow_width=640, scene_cut_threshold=scene_cut_threshold)


def _plan(multiplier: int, native_multiplier: int) -> dict[str, Any]:
    multiplier = int(multiplier)
    if multiplier < 2 or multiplier > 6:
        raise ValueError("Native MFG multiplier must be between 2x and 6x.")
    if native_multiplier >= multiplier:
        return {
            "path": "native_exact",
            "stage_generated_counts": [multiplier - 1],
            "grid_multiplier": multiplier,
        }
    if native_multiplier < 2:
        raise NativeDLSSGError("Native DLSSG runtime does not expose even a 2x Frame Generation path.")
    stages = int(math.ceil(math.log2(multiplier)))
    grid = 1 << stages
    return {
        "path": "cascade",
        "stage_generated_counts": [1] * stages,
        "grid_multiplier": grid,
    }


def stream_native_mfg(
    images: torch.Tensor,
    *,
    source_frame_rate: float,
    multiplier: int,
    emit: Callable[[np.ndarray, int, Fraction, TimedFrame], None],
    guide_source: str = "internal_dis",
    motion: MotionPacket | None = None,
    scene_cut_threshold: float = 0.24,
    auto_bootstrap: bool = True,
    session_factory: Callable[..., DirectDLSSGSession] = DirectDLSSGSession,
) -> dict[str, Any]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(f"Expected IMAGE [T,H,W,C], got {tuple(images.shape)}")
    n, h, w, _c = [int(x) for x in images.shape]
    if n < 1:
        raise ValueError("Native MFG received an empty IMAGE batch.")
    source_rate = Fraction(str(float(source_frame_rate))).limit_denominator(1_000_000)
    if source_rate <= 0:
        raise ValueError("source_frame_rate must be > 0")

    capabilities = probe_native_capabilities(auto_bootstrap=auto_bootstrap)
    if not capabilities.get("available", False):
        raise NativeDLSSGError(
            "Native NVIDIA DLSS Frame Generation is unavailable: "
            + str(capabilities.get("detail") or capabilities)
        )
    native_multiplier = int(capabilities.get("native_multiplier", 1))
    plan = _plan(int(multiplier), native_multiplier)
    target_rate = source_rate * int(multiplier)
    output_count = (n - 1) * int(multiplier) + 1

    sessions: list[DirectDLSSGSession] = []
    stages: list[NativeStage] = []
    writer = NearestTimelineWriter(target_rate, output_count, emit)
    try:
        stage_input_frames = n
        for stage_index, generated_count in enumerate(plan["stage_generated_counts"]):
            session = session_factory(
                w,
                h,
                max(1, int(stage_input_frames)),
                int(generated_count),
            )
            sessions.append(session)
            if stage_index == 0:
                guide = _make_guide(guide_source, w, h, motion, scene_cut_threshold)
            else:
                guide = DISGuideGenerator(w, h, flow_width=640, scene_cut_threshold=scene_cut_threshold)
            stages.append(
                NativeStage(
                    session,
                    w,
                    h,
                    int(generated_count),
                    guide,
                    detect_source_cuts=(stage_index == 0),
                )
            )
            stage_input_frames = max(1, (stage_input_frames - 1) * (int(generated_count) + 1) + 1)

        segment = 0
        for source_index in range(n):
            # Respect ComfyUI cancellation when available.
            try:
                import comfy.model_management as mm  # type: ignore

                mm.throw_exception_if_processing_interrupted()
            except ImportError:
                pass
            rgba = _frame_to_rgba8(images[source_index])
            timed = TimedFrame(
                rgba=rgba,
                timestamp=Fraction(source_index, 1) / source_rate,
                segment=segment,
                provenance="Source",
                source_index=source_index,
            )
            items = [timed]
            for stage in stages:
                next_items: list[TimedFrame] = []
                for item in items:
                    next_items.extend(stage.push(item))
                items = next_items
            for item in items:
                writer.push(item)
        writer.finish()
    finally:
        worker_logs = [session.log_text() for session in sessions]
        for session in sessions:
            try:
                session.close()
            except Exception:
                pass

    return {
        "backend": "native_dlssg",
        "path": plan["path"],
        "source_frame_rate": float(source_rate),
        "target_frame_rate": float(target_rate),
        "multiplier": int(multiplier),
        "input_frames": n,
        "output_frames": output_count,
        "native_multiplier_max": native_multiplier,
        "native_generated_frame_max": int(capabilities.get("multi_frame_count_max", 0)),
        "stage_generated_counts": list(plan["stage_generated_counts"]),
        "grid_multiplier": int(plan["grid_multiplier"]),
        "guide_source": str(guide_source),
        "scene_cut_threshold": float(scene_cut_threshold),
        "selected_generated_frames": int(writer.generated_selected),
        "selected_source_frames": int(writer.source_selected),
        "maximum_temporal_error_seconds": float(writer.max_error),
        "stage_scene_cuts": [int(stage.scene_cuts) for stage in stages],
        "stage_duplicates": [int(stage.duplicates) for stage in stages],
        "stage_resets": [int(stage.resets) for stage in stages],
        "stage_generated_frames": [int(stage.generated) for stage in stages],
        "capabilities": capabilities,
        "worker_logs": worker_logs,
    }


def generate_native_mfg_images(
    images: torch.Tensor,
    *,
    source_frame_rate: float,
    multiplier: int,
    guide_source: str = "internal_dis",
    motion: MotionPacket | None = None,
    scene_cut_threshold: float = 0.24,
    auto_bootstrap: bool = True,
    output_precision: str = "auto",
    output_storage: str = "auto",
    clean_cache: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(f"Expected IMAGE [T,H,W,C], got {tuple(images.shape)}")
    n, h, w, c = [int(x) for x in images.shape]
    out_channels = 4 if c == 4 else 3
    output_frames = (n - 1) * int(multiplier) + 1
    out_shape = (output_frames, h, w, out_channels)
    out_dtype = resolve_dtype(
        requested=str(output_precision),
        shape=out_shape,
        input_dtype=images.dtype,
        auto_fp16_threshold_mb=512,
    )
    estimated_bytes = estimate_bytes(out_shape, out_dtype)
    max_image_gib = float(os.environ.get("AETHERSCALE_MFG_IMAGE_MAX_GIB", "64"))
    max_image_bytes = int(max_image_gib * (1024 ** 3))
    if estimated_bytes > max_image_bytes:
        raise RuntimeError(
            "Native MFG IMAGE output is too large to materialize safely as a ComfyUI IMAGE batch. "
            f"This request would require {estimated_bytes / (1024 ** 3):.2f} GiB "
            f"({output_frames} frames at {w}x{h}, {str(out_dtype).replace('torch.', '')}). "
            "Use AetherScale • MFG Video after all spatial/HDR processing to stream generated "
            "frames directly into the encoder without creating the multiplied IMAGE batch. "
            "For deliberate huge IMAGE materialization, set AETHERSCALE_MFG_IMAGE_MAX_GIB to a larger value."
        )
    out, storage = allocate_cpu_tensor(
        out_shape,
        dtype=out_dtype,
        storage_mode=str(output_storage),
        prefix="mfg_native",
        mmap_threshold_mb=512,
        clean_cache=bool(clean_cache),
    )
    console_progress = ConsoleProgress("MFG / native DLSSG", output_frames, unit="frame")

    def emit(rgba: np.ndarray, index: int, ideal: Fraction, frame: TimedFrame) -> None:
        del ideal, frame
        arr = rgba if out_channels == 4 else rgba[..., :3]
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32).mul_(1.0 / 255.0)
        if out_dtype != torch.float32:
            tensor = tensor.to(out_dtype)
        out[index].copy_(tensor)
        sync_file_backed_tensor(out, bytes_written=tensor.numel() * tensor.element_size())
        console_progress.update(1)

    stats = stream_native_mfg(
        images,
        source_frame_rate=float(source_frame_rate),
        multiplier=int(multiplier),
        emit=emit,
        guide_source=str(guide_source),
        motion=motion,
        scene_cut_threshold=float(scene_cut_threshold),
        auto_bootstrap=bool(auto_bootstrap),
    )
    sync_file_backed_tensor(out, force=True)
    console_progress.close(status="done")
    stats.update(
        {
            "output_dtype": str(out_dtype).replace("torch.", ""),
            "output_storage_backend": storage.backend,
            "output_storage_bytes": int(storage.bytes),
            "output_storage_gib": round(storage.bytes / (1024**3), 3),
            "output_storage_path": storage.path,
            "clean_cache": bool(clean_cache),
        }
    )
    return out, stats
