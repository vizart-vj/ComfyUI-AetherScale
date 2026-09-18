from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .mfg import _find_ffmpeg
from .progress import ConsoleProgress, throw_if_interrupted
from .storage import CACHE_DIR, map_existing_file_tensor, make_cache_file_path

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}


class VideoLoadError(RuntimeError):
    pass


def list_input_videos() -> list[str]:
    try:
        import folder_paths  # type: ignore
        root = Path(folder_paths.get_input_directory()).resolve()
    except Exception:
        return ["<use path_override>"]
    out: list[str] = []
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
                out.append(p.relative_to(root).as_posix())
    except OSError:
        pass
    return sorted(out) or ["<use path_override>"]


def resolve_video_path(video: str, path_override: str = "") -> Path:
    override = str(path_override or "").strip().strip('"')
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p.resolve()
        raise VideoLoadError(f"Video path does not exist: {p}")

    name = str(video or "").strip()
    if not name or name == "<use path_override>":
        raise VideoLoadError("Choose a video from ComfyUI/input or set path_override.")
    p = Path(name)
    if p.is_file():
        return p.resolve()
    try:
        import folder_paths  # type: ignore
        # get_annotated_filepath supports normal input names in current ComfyUI.
        try:
            candidate = Path(folder_paths.get_annotated_filepath(name))
            if candidate.is_file():
                return candidate.resolve()
        except Exception:
            pass
        candidate = Path(folder_paths.get_input_directory()) / name
        if candidate.is_file():
            return candidate.resolve()
    except Exception:
        pass
    raise VideoLoadError(f"Video file not found: {name}")


def _find_ffprobe(ffmpeg: str) -> str:
    ffmpeg_path = Path(ffmpeg)
    names = ["ffprobe.exe", "ffprobe"] if os.name == "nt" else ["ffprobe", "ffprobe.exe"]
    for name in names:
        p = ffmpeg_path.with_name(name)
        if p.is_file():
            return str(p)
    found = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if found:
        return found
    raise VideoLoadError(
        "ffprobe was not found next to FFmpeg or in PATH. Install/enable VideoHelperSuite "
        "or put ffprobe in PATH."
    )


def _ratio(value: Any, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return float(default)
    try:
        if "/" in text:
            a, b = text.split("/", 1)
            b_f = float(b)
            return float(a) / b_f if b_f else float(default)
        return float(text)
    except Exception:
        return float(default)


def probe_video(path: Path, ffmpeg: str | None = None) -> dict[str, Any]:
    ffmpeg = ffmpeg or _find_ffmpeg()
    ffprobe = _find_ffprobe(ffmpeg)
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if cp.returncode != 0:
        detail = cp.stderr.decode("utf-8", errors="replace")[-3000:]
        raise VideoLoadError(f"ffprobe failed for {path}:\n{detail}")
    try:
        payload = json.loads(cp.stdout.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise VideoLoadError(f"Could not parse ffprobe JSON: {exc}") from exc

    streams = payload.get("streams") if isinstance(payload, dict) else None
    streams = streams if isinstance(streams, list) else []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        raise VideoLoadError(f"No video stream found in {path}")
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _ratio(video_stream.get("avg_frame_rate"), _ratio(video_stream.get("r_frame_rate"), 0.0))
    duration = _ratio(video_stream.get("duration"), 0.0)
    if duration <= 0:
        fmt = payload.get("format") if isinstance(payload, dict) else None
        duration = _ratio(fmt.get("duration") if isinstance(fmt, dict) else 0.0, 0.0)
    nb_frames = video_stream.get("nb_frames")
    try:
        frame_count = int(nb_frames) if nb_frames not in (None, "N/A", "") else 0
    except Exception:
        frame_count = 0
    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = max(1, int(round(duration * fps)))

    audio = None
    if isinstance(audio_stream, dict):
        try:
            sample_rate = int(audio_stream.get("sample_rate") or 48000)
        except Exception:
            sample_rate = 48000
        try:
            channels = int(audio_stream.get("channels") or 2)
        except Exception:
            channels = 2
        audio = {
            "sample_rate": max(8000, sample_rate),
            "channels": max(1, min(8, channels)),
            "codec": str(audio_stream.get("codec_name") or ""),
        }

    return {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": int(frame_count),
        "codec": str(video_stream.get("codec_name") or ""),
        "pix_fmt": str(video_stream.get("pix_fmt") or ""),
        "audio": audio,
    }


def _read_audio(path: Path, ffmpeg: str, meta: dict[str, Any], start_time: float, duration: float | None) -> dict[str, Any]:
    audio_meta = meta.get("audio")
    if not isinstance(audio_meta, dict):
        return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32), "sample_rate": 48000}
    sr = int(audio_meta.get("sample_rate") or 48000)
    channels = int(audio_meta.get("channels") or 2)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_time > 0:
        cmd += ["-ss", f"{start_time:.9f}"]
    cmd += ["-i", str(path), "-map", "0:a:0", "-vn"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.9f}"]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "pipe:1"]
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    throw_if_interrupted()
    if cp.returncode != 0:
        detail = cp.stderr.decode("utf-8", errors="replace")[-2000:]
        print(f"[AetherScale] Video Loader audio decode skipped: {detail}", flush=True)
        return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32), "sample_rate": sr}
    arr = np.frombuffer(cp.stdout, dtype=np.float32)
    usable = (arr.size // channels) * channels
    if usable <= 0:
        return {"waveform": torch.zeros((1, channels, 0), dtype=torch.float32), "sample_rate": sr}
    arr = arr[:usable].reshape(-1, channels)
    waveform = torch.from_numpy(arr.copy()).transpose(0, 1).unsqueeze(0).contiguous()
    return {"waveform": waveform, "sample_rate": sr}


def _estimated_frames(meta: dict[str, Any], start_time: float, frame_load_cap: int, force_rate: float) -> int:
    duration = max(0.0, float(meta.get("duration") or 0.0) - max(0.0, float(start_time)))
    fps = float(force_rate) if force_rate > 0 else float(meta.get("fps") or 0.0)
    n = int(round(duration * fps)) if duration > 0 and fps > 0 else int(meta.get("frame_count") or 0)
    if frame_load_cap > 0:
        n = min(n if n > 0 else frame_load_cap, frame_load_cap)
    return max(0, n)


def _choose_dtype(precision: str, estimated_frames: int, width: int, height: int) -> tuple[torch.dtype, np.dtype]:
    p = str(precision)
    if p == "float16":
        return torch.float16, np.dtype(np.float16)
    if p == "float32":
        return torch.float32, np.dtype(np.float32)
    estimated_f32 = max(1, estimated_frames) * max(1, width) * max(1, height) * 3 * 4
    if estimated_f32 >= 768 * 1024 * 1024:
        return torch.float16, np.dtype(np.float16)
    return torch.float32, np.dtype(np.float32)


def _video_flush_threshold_bytes() -> int:
    raw = str(os.environ.get("AETHERSCALE_VIDEO_LOAD_FLUSH_MB", os.environ.get("AETHERSCALE_SPILL_FLUSH_MB", "128"))).strip()
    try:
        mb = max(16, int(float(raw)))
    except Exception:
        mb = 128
    return mb * 1024 * 1024


def load_video_low_ram(
    *,
    video: str,
    path_override: str = "",
    force_rate: float = 0.0,
    start_time: float = 0.0,
    frame_load_cap: int = 0,
    precision: str = "auto",
    decode_chunk_frames: int = 4,
    load_audio: bool = True,
    clean_cache: bool = True,
) -> tuple[torch.Tensor, int, dict[str, Any], dict[str, Any], float, str]:
    path = resolve_video_path(video, path_override)
    ffmpeg = _find_ffmpeg()
    meta = probe_video(path, ffmpeg)
    width = int(meta["width"])
    height = int(meta["height"])
    if width <= 0 or height <= 0:
        raise VideoLoadError(f"Invalid video dimensions reported by ffprobe: {width}x{height}")

    source_fps = float(meta.get("fps") or 0.0)
    target_fps = float(force_rate) if float(force_rate) > 0 else source_fps
    if target_fps <= 0:
        raise VideoLoadError("Could not determine video frame rate; set force_rate explicitly.")
    estimated = _estimated_frames(meta, float(start_time), int(frame_load_cap), float(force_rate))
    torch_dtype, np_dtype = _choose_dtype(str(precision), estimated, width, height)
    element_size = 2 if torch_dtype == torch.float16 else 4
    estimated_bytes = estimated * width * height * 3 * element_size if estimated > 0 else 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(CACHE_DIR).free
    margin = max(2 * 1024**3, min(8 * 1024**3, int(max(estimated_bytes, 1) * 0.05)))
    if estimated_bytes and disk_free < estimated_bytes + margin:
        raise VideoLoadError(
            f"Low-RAM Video Loader needs about {estimated_bytes / 2**30:.2f} GiB for the IMAGE backing file, "
            f"but only {disk_free / 2**30:.2f} GiB is free in {CACHE_DIR}. "
            "Set AETHERSCALE_CACHE_DIR to a drive with more free space."
        )

    out_path = make_cache_file_path("video_load", auto_delete=bool(clean_cache))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if float(start_time) > 0:
        cmd += ["-ss", f"{float(start_time):.9f}"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-an"]
    if float(force_rate) > 0:
        cmd += ["-vf", f"fps={float(force_rate):.12g}"]
    if int(frame_load_cap) > 0:
        cmd += ["-frames:v", str(int(frame_load_cap))]
    cmd += ["-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]

    frame_bytes = width * height * 3
    chunk_frames = max(1, min(32, int(decode_chunk_frames)))
    chunk_bytes = frame_bytes * chunk_frames
    scale = np_dtype.type(1.0 / 255.0)
    count = 0
    progress = ConsoleProgress("Video Loader / decode", estimated, unit="frame")
    stderr_tail = b""
    proc: subprocess.Popen | None = None
    flush_threshold = _video_flush_threshold_bytes()
    bytes_since_flush = 0
    t0 = time.perf_counter()
    try:
        with open(out_path, "wb", buffering=8 * 1024 * 1024) as fout:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.stdout is None or proc.stderr is None:
                raise VideoLoadError("Failed to open FFmpeg decode pipes.")
            pending = bytearray()
            while True:
                throw_if_interrupted()
                data = proc.stdout.read(chunk_bytes - len(pending))
                if data:
                    pending.extend(data)
                if not data or len(pending) >= chunk_bytes:
                    usable = (len(pending) // frame_bytes) * frame_bytes
                    if usable:
                        raw = np.frombuffer(memoryview(pending)[:usable], dtype=np.uint8)
                        n = usable // frame_bytes
                        raw = raw.reshape(n, height, width, 3)
                        converted = raw.astype(np_dtype, copy=True)
                        converted *= scale
                        written = fout.write(memoryview(converted).cast("B"))
                        bytes_since_flush += int(written)
                        if bytes_since_flush >= flush_threshold:
                            throw_if_interrupted()
                            fout.flush()
                            os.fsync(fout.fileno())
                            bytes_since_flush = 0
                            throw_if_interrupted()
                        count += n
                        progress.set(count)
                        del converted, raw
                        if usable == len(pending):
                            pending.clear()
                        else:
                            pending = bytearray(pending[usable:])
                    if not data:
                        break
            # Capture diagnostics only after stdout is drained to avoid pipe deadlock.
            stderr_tail = proc.stderr.read()[-4000:]
            rc = proc.wait()
            if rc != 0:
                detail = stderr_tail.decode("utf-8", errors="replace")
                raise VideoLoadError(f"FFmpeg decode failed with exit code {rc}:\n{detail}")
            fout.flush()
            os.fsync(fout.fileno())
        throw_if_interrupted()
        if count <= 0:
            raise VideoLoadError("Video decoder produced no frames.")
        expected_size = count * height * width * 3 * element_size
        actual_size = out_path.stat().st_size
        if actual_size != expected_size:
            raise VideoLoadError(
                f"Decoded backing file has unexpected size: {actual_size} bytes, expected {expected_size}."
            )
        images, storage = map_existing_file_tensor(
            out_path,
            (count, height, width, 3),
            dtype=torch_dtype,
            auto_delete=bool(clean_cache),
            backend="stream_decode_mmap",
        )
        progress.set(count)
        progress.close()
    except BaseException:
        try:
            progress.close()
        except Exception:
            pass
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    loaded_duration = count / target_fps
    audio_duration = loaded_duration if int(frame_load_cap) > 0 or float(force_rate) > 0 else None
    audio = _read_audio(path, ffmpeg, meta, float(start_time), audio_duration) if bool(load_audio) else {
        "waveform": torch.zeros((1, 1, 0), dtype=torch.float32), "sample_rate": 48000
    }
    video_info = {
        "source_fps": float(source_fps),
        "source_frame_count": int(meta.get("frame_count") or 0),
        "source_duration": float(meta.get("duration") or 0.0),
        "source_width": int(width),
        "source_height": int(height),
        "loaded_fps": float(target_fps),
        "loaded_frame_count": int(count),
        "loaded_duration": float(loaded_duration),
        "loaded_width": int(width),
        "loaded_height": int(height),
    }
    stats = {
        "engine": "FFmpeg chunked rawvideo -> sequential FP16/FP32 backing file",
        "source": str(path),
        "source_width": width,
        "source_height": height,
        "source_fps": source_fps,
        "source_duration": float(meta.get("duration") or 0.0),
        "loaded_frame_count": int(count),
        "loaded_fps": float(target_fps),
        "loaded_duration": float(loaded_duration),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "storage_backend": storage.backend,
        "storage_path": storage.path,
        "storage_bytes": int(storage.bytes),
        "storage_gib": round(storage.bytes / 2**30, 3),
        "decode_chunk_frames": int(chunk_frames),
        "estimated_decode_working_set_mb": round(chunk_frames * width * height * 3 * (1 + element_size) / 2**20, 2),
        "sequential_flush_mb": int(flush_threshold // 2**20),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "load_audio": bool(load_audio),
    }
    return images, int(count), audio, video_info, float(target_fps), json.dumps(stats, indent=2)
