[![AetherScale screenshot](ex.png)](ex.png)

# AetherScale for ComfyUI

**GPU-native NVIDIA video enhancement, restoration, temporal analysis, experimental DLSS 5 Neural Rendering, native DLSS Frame Generation, and high-throughput NVENC/ProRes video output for ComfyUI.**

AetherScale is a Windows/NVIDIA-focused custom node suite for high-quality image and video enhancement with practical long-video memory handling. It combines NVIDIA VFX processing, a CUDA-native HDR-style enhancer, temporal motion analysis, memory-mapped long-video storage, in-process DLSS 5 Neural Rendering, native DLSS Frame Generation, and production-oriented video output in one node pack.

**Author:** noise  
**Current version:** 0.9.2  
**ComfyUI folder:** `ComfyUI-AetherScale`




## What's new in v0.9.2

- Fixed the **Video Combine node that kept growing vertically** after every execution or UI interaction.
- The preview now exists as one persistent LiteGraph DOM widget. Its height is calculated absolutely from the current node width and the real video aspect ratio; it is never added to the node's previous height.
- Re-running Combine only swaps the video source. It does not remove/recreate the player or accumulate layout state.
- Old workflows saved with an abnormally tall Combine node are automatically re-fitted once after loading.
- The implementation uses AetherScale-owned frontend code; VHS Video Combine was inspected only as a behavior reference for stable preview sizing.

## What's new in v0.9.1

- Added **AetherScale • Video Loader**, a low-RAM replacement for whole-batch IMAGE video loaders in long post-production workflows.
- The loader uses FFmpeg raw-video decode in small chunks and writes each chunk **sequentially** into an AetherScale backing file instead of first constructing a giant Python/NumPy float32 batch.
- `precision = auto` selects FP16 for large videos. A 15-second 2464×512@24 source is about **5.08 GiB as RGB float32** but only **2.54 GiB as RGB FP16 backing storage**, while working RAM stays bounded by the decode chunk rather than the whole sequence.
- The finished backing file is mapped only after decoding is complete, and sequential writes are forced to disk in bounded 128 MiB windows, so Windows does not accumulate tens of GiB of dirty cache/mmap pages during video loading.
- The loader returns `IMAGE`, `frame_count`, `AUDIO`, a `VHS_VIDEOINFO`-compatible dictionary, direct `frame_rate`, and JSON stats, so it can replace a VHS loader with minimal rewiring.
- Added early disk-space preflight, ComfyUI interrupt polling, decode progress, optional audio extraction, `start_time`, `frame_load_cap`, and `force_rate` controls.
- Existing AetherScale node schemas remain compatible; the new loader is additive.
- README, changelog, Registry/package metadata, runtime User-Agent, diagnostics, and release notes are synchronized for v0.9.1.

## Features

- NVIDIA VFX Video Super Resolution
- artifact reduction, denoise, and deblur workflows
- built-in CUDA HDR-style tone/color enhancement
- streaming, frame-by-frame processing to reduce peak VRAM/RAM pressure
- low-RAM FFmpeg video loading into FP16/FP32 file-backed IMAGE tensors
- adaptive long-video storage: anonymous/pagefile-backed mappings with bounded-writeback disk-spill fallback when Windows commit is insufficient
- `clean_cache=true` normally stays fileless, but can temporarily spill to disk instead of failing when Windows cannot reserve enough commit
- temporal motion analysis with scene-cut detection
- compact FP16 motion storage for long sequences
- in-process DLSS 5 Neural Rendering with CUDA/D3D12 interop and carrier fallback
- DLSS output modes from native 1x through 3x
- automatic runtime discovery/bootstrap with pinned sources and checksum verification
- dedicated diagnostics and runtime-management nodes
- native NVIDIA DLSS Frame Generation backend with direct generated-frame readback
- native DLSSG-to-FFmpeg/NVENC streaming output with no multiplied IMAGE batch
- high-throughput generic IMAGE-to-video output with chunked NVENC or high-depth ProRes encoding
- direct MFG-to-FFmpeg/NVENC streaming output for long 2x/3x/4x exports

## Nodes

| Node | Purpose |
| --- | --- |
| **AetherScale • Super Resolution** | NVIDIA VFX upscaling with streaming/mmap long-video output handling |
| **AetherScale • Restoration** | Artifact reduction, denoise, and deblur |
| **AetherScale • HDR** | CUDA-native HDR-style tone/color enhancement with future native-VFX auto-detection |
| **AetherScale • Motion Analysis** | Current-to-previous motion estimation and scene-cut detection |
| **AetherScale • Neural Rendering** | In-process DLSS 5 Neural Rendering with CUDA/D3D12 interop, chunking, and explicit fallbacks |
| **AetherScale • Neural VRAM Planner** | Memory planning for Neural Rendering workloads |
| **AetherScale • Runtime** | Inspect, bootstrap, repair, or clear private runtimes |
| **AetherScale • Diagnostics** | GPU, CUDA, runtime, and capability reporting |
| **AetherScale • MFG** | Native NVIDIA DLSS Frame Generation to IMAGE, with surrogate fallback/debug mode |
| **AetherScale • MFG Video** | Native DLSSG streamed directly to FFmpeg/NVENC without creating a giant IMAGE batch |
| **AetherScale • Video Loader** | Low-RAM FFmpeg loader: chunked decode directly into a file-backed IMAGE tensor, plus FPS/audio/stats outputs |
| **AetherScale • Video Combine** | Fast generic IMAGE + optional AUDIO to MP4/MKV/MOV with NVENC, x264, or ProRes |


## Console progress reporting

AetherScale v0.9.2 uses `tqdm` for the terminal row and ComfyUI's native `ProgressBar` for browser progress and interruption. Each long-running stage owns one live progress bar instead of printing a new line for every refresh. Example:

```text
[AetherScale] Super Resolution |██████████▌             | 41.67% | 250/600 frames | 2.31 frame/s | ETA 02:31
```

The bar dynamically adapts to the current console width and is redrawn in place. When the stage completes, the final 100% state remains as one history line and the normal node timing line follows. If another subsystem prints while a bar is active, `tqdm` handles the redraw using the same behavior users already see from native ComfyUI progress bars.

### Windows disk-spill writeback

Large file-backed mappings are now flushed with bounded backpressure rather than leaving tens of GiB of dirty mapped pages for the Windows Cache Manager to drain later. The default dirty window is 128 MiB and can be changed with `AETHERSCALE_SPILL_FLUSH_MB`. This makes progress reflect real storage throughput and prevents a killed ComfyUI process from leaving minutes of heavy background writeback.

For extremely large MFG outputs, prefer `AetherScale • MFG Video`. A 2x/4x IMAGE materialization at multi-megapixel resolutions can exceed tens or hundreds of GiB even in FP16; direct MFG Video avoids that multiplied tensor entirely.

## Requirements

- Windows 10/11 64-bit
- NVIDIA RTX GPU
- current NVIDIA display driver
- ComfyUI with Python 3.10+
- internet access on first use for optional runtime bootstrap components, or the corresponding official pinned archive available locally

The experimental DLSS 5 path is hardware/driver/runtime dependent. RTX 50-series hardware is the primary target for the stock Neural Rendering runtime; compatibility of other generations depends on the selected runtime path.

## Installation

### ComfyUI Manager / Registry

Once published to the Comfy Registry, search for **AetherScale** in ComfyUI Manager and install it normally.

### Git

Clone into `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/vizart-vj/ComfyUI-AetherScale.git
```

Then restart ComfyUI.

### Manual

Extract the folder so the final path is:

```text
ComfyUI/custom_nodes/ComfyUI-AetherScale/
```

The root folder name is intentionally stable and must remain `ComfyUI-AetherScale`.

## Quick start

For video files, use **AetherScale • Video Loader** before the enhancement chain when the source is more than a small clip. Its default `precision = auto`, `decode_chunk_frames = 4`, and `clean_cache = true` avoid the large RAM spike caused by materializing the whole decoded sequence as one float32 array. Its first four outputs intentionally mirror the common VHS loader shape (`IMAGE`, frame count, `AUDIO`, `VHS_VIDEOINFO`); it also exposes `frame_rate` directly for **AetherScale • Video Combine**.

For conventional upscaling, start with **AetherScale • Super Resolution** and use the automatic memory controls.

For DLSS 5 Neural Rendering:

1. connect the image/video frame batch to **AetherScale • Motion Analysis**;
2. keep `motion_mode = compact_flow` for long sequences;
3. connect its motion output to **AetherScale • Neural Rendering**;
4. use `backend = native_interop`;
5. start with `upscale_mode = native_1x`; use the separate **AetherScale • Super Resolution** node when you want true NVIDIA VFX upscaling before/after NR.

Recommended native settings for a 16 GB RTX 50-series GPU:

```text
backend = native_interop
native_chunk_frames = 8
native_fallback = carrier
output_precision = auto
output_storage = auto
clean_cache = true
```

`native_chunk_frames` is a ceiling, not a promise: AetherScale shrinks the current chunk automatically when free VRAM is lower than expected. The connected AetherScale motion packet supplies scene-cut reset timing; its optical-flow tensor is not copied into the native NR bridge unless a future runtime exposes a stable motion-vector contract.

## Low-RAM video input

`AetherScale • Video Loader` exists because memory-safe processing cannot repair an upstream loader after that loader has already materialized the complete video in RAM. It decodes the source with FFmpeg in small RGB24 chunks, converts only the current chunk to the selected ComfyUI precision, writes it sequentially to the AetherScale cache drive, and then maps the completed file as an `IMAGE` tensor.

Recommended defaults:

```text
precision = auto
decode_chunk_frames = 4
force_rate = 0        # preserve source FPS
frame_load_cap = 0    # full source
load_audio = true
clean_cache = true
```

`path_override` accepts an absolute file path. Otherwise choose a video already present under `ComfyUI/input`. Set `AETHERSCALE_CACHE_DIR` before launching ComfyUI if the backing tensor should live on another SSD/NVMe drive.

The loader deliberately uses sequential file writes rather than a writable mmap during decode. By default it flushes every 128 MiB (`AETHERSCALE_VIDEO_LOAD_FLUSH_MB`, falling back to `AETHERSCALE_SPILL_FLUSH_MB`) so the Windows file cache cannot turn the whole output into dirty resident pages while frames are arriving.

## Long-video memory architecture

AetherScale avoids moving an entire video batch to CUDA when the operation can be streamed. The main enhancement paths process frames incrementally and large outputs can use FP16 plus mmap-backed CPU storage.

### `clean_cache`

`clean_cache` is available on large-output **Super Resolution** and **Neural Rendering** paths.

- `true` — recommended/default. AetherScale first uses an anonymous/pagefile-backed mapping. If Windows commit is too low, it automatically switches to a temporary disk-backed spill mmap instead of failing with `WinError 1450`.
- `false` — uses a traditional persistent disk-backed `.mmap` in `.aetherscale_cache` for debugging or workflows where persistent backing storage is specifically desired.

Temporary clean-cache spill mappings are marked for automatic deletion when ComfyUI releases the output tensor. If ComfyUI exits or crashes while one is still mapped, the existing PID-aware orphan cleanup removes it on the next AetherScale cache operation/startup. Set `AETHERSCALE_CACHE_DIR` before launching ComfyUI to place spill/cache mappings on another drive.

This keeps the streaming/low-working-set architecture while avoiding Windows commit-limit crashes. Downstream ComfyUI nodes can still materialize or copy a full IMAGE batch, so extremely long/high-resolution workflows should remain memory-conscious.

## HDR backend

Current NVIDIA Video Effects SDK releases do not expose a public HDR effect. AetherScale therefore uses its built-in CUDA HDR-style enhancer while preserving the existing HDR node controls:

- profile (`balanced / cinematic / punchy / natural`)
- strength
- saturation
- contrast
- highlight preservation

If a future NVIDIA VFX runtime exposes a compatible HDR effect, AetherScale can select it automatically. The node outputs normalized ComfyUI IMAGE tensors; it performs HDR-style tone/color enhancement and does not attach HDR10/PQ mastering metadata.

## Runtime bootstrap and security

AetherScale does **not** store downloaded runtime binaries in the Git repository. Runtime/cache directories are ignored by Git.

Depending on the selected node/backend, AetherScale may use or bootstrap:

- `nvidia-vfx==0.1.0.1` for NVIDIA VFX processing;
- a pinned MIT native bridge/caller from `orex2121/ComfyUI-DLSS5-orex` for the default in-process NR backend;
- the pinned `310.8.SF-v2` Neural Rendering runtime verified for that bridge (SHA-256 `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`) when native NR auto-bootstrap is enabled;
- the pinned `Merserk/dlss5-visual-enhancer` v1.0 portable release for the experimental carrier backend;
- the pinned MIT bridge/caller from `lisitskyaa/ComfyUI-DLSS5-NR` for the legacy direct diagnostic backend;
- selected DLSSNR runtime packages for the legacy direct backend.

Pinned archives are verified against hard-coded SHA-256 values before use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact sources and hashes.

Carrier bootstrap is network-resilient on Windows. AetherScale first attempts a direct connection that bypasses stale system proxies, then retries through the configured proxy, `curl`, and PowerShell. If GitHub is unavailable on the machine, an already downloaded official `DLSS.5.Visual.Enhancer.v1.0.zip` is detected automatically in the Windows `Downloads` folder. Advanced users can set `AETHERSCALE_CARRIER_ARCHIVE` to a local copy or `AETHERSCALE_CARRIER_URL` to a mirror; the pinned SHA-256 is always checked before extraction.

The **`native_interop`** backend is the default Neural Rendering path. `carrier` remains an explicit compatibility fallback; `legacy_direct` remains diagnostic/reproducibility only.

## Native Neural Rendering architecture

`native_interop` intentionally separates AetherScale's code from the third-party native bridge:

```text
ComfyUI IMAGE / AetherScale chunk scheduler
        ↓
selected Torch CUDA device
        ↓ raw device pointers when supported
AetherScale native ABI adapter
        ↓
third-party MIT D3D12/NGX bridge
        ↓
DLSS Neural Rendering feature 18
        ↓
AetherScale output policy
  ├─ direct GPU result when safely small
  └─ FP16 RAM / anonymous map / disk spill for long video
```

AetherScale does **not** copy the OreX Python node implementation. The pinned bridge and caller binaries are used as declared third-party MIT dependencies, with the upstream license shipped in `third_party/ComfyUI-DLSS5-OreX-LICENSE.txt`. AetherScale owns the surrounding adapter, multi-GPU matching, chunk scheduler, progress/interrupt behavior, scene reset integration, storage policy, and fallbacks. See `native/NR_INTEROP_DESIGN.md`.

The optional `upscale_mode` values above `native_1x` are retained for workflow compatibility. In `native_interop` they use a GPU bicubic pre-resize followed by Neural Rendering; that is **not** DLSS Super Resolution. For true upscale processing use **AetherScale • Super Resolution**.

## GPU selection

The NVIDIA VFX/CUDA paths use CUDA device selection. The carrier backend uses D3D12/DXGI and therefore follows Windows graphics adapter routing rather than PyTorch CUDA indexing. AetherScale applies a per-application Windows **High Performance** GPU preference to the carrier worker and reports the expected adapter in node statistics.

`AetherScale • Video Combine` uses a separate NVENC routing layer. Its `nvenc_gpu` choices are physical GPUs discovered through `nvidia-smi`; the selected physical adapter is isolated for the FFmpeg child process and remapped to FFmpeg logical CUDA GPU 0. This avoids the common multi-GPU mismatch where ComfyUI/PyTorch sees a restricted CUDA device list but `nvidia-smi` still reports physical ordinals for all installed GPUs.

## Development status

The VFX enhancement nodes are the stable portion of the project. DLSS 5 Neural Rendering remains **experimental** and is expected to evolve as public runtime behavior, drivers, and community implementations mature.

When reporting a Neural Rendering issue, include:

- GPU model(s)
- NVIDIA driver version
- AetherScale version
- full ComfyUI traceback
- AetherScale `stats` output when available

## Third-party software

AetherScale is an independent community project and is not affiliated with or endorsed by NVIDIA, Topaz Labs, RenoDX, ReShade, or the referenced third-party projects.

AetherScale source code is licensed under the MIT License. Third-party components retain their own licenses and terms. No third-party license is replaced or relicensed by this repository.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT License — Copyright (c) 2026 **noise**.



## MFG Video fast path

For long 2x/3x/4x jobs, use **AetherScale • MFG Video** instead of sending the multiplied MFG IMAGE batch into a generic video-combine node.

The fast path is:

```text
source IMAGE batch
      ↓
MFG frame generation
      ↓ one frame at a time
RGB24 quantization
      ↓
FFmpeg / NVENC
      ↓
MP4 file
```

The multiplied frame sequence is never materialized as a complete ComfyUI IMAGE tensor. This avoids writing tens of gigabytes of generated frames to RAM/pagefile and immediately reading them back for encoding.

Connect the same source FPS value you would normally send to Video Combine into `source_frame_rate`. AetherScale automatically encodes at `source_frame_rate × multiplier`, so 2x/3x/4x preserves source duration.

Use **AetherScale • MFG** instead when you need native generated frames as an IMAGE output for inspection or additional image-domain nodes.

## Video Combine

`AetherScale • Video Combine` is the generic fast output node for ordinary ComfyUI `IMAGE` batches, including outputs from Super Resolution, Neural Rendering, native MFG, RIFE, or any other image-producing node.

The performance path is intentionally different from frame-by-frame Python encoders:

1. several BHWC frames are packed to contiguous RGB8 in one Torch operation;
2. the packed chunk is passed to FFmpeg through the Python buffer protocol, avoiding a separate `.tobytes()` allocation;
3. a small producer thread prepares the next chunk while FFmpeg/NVENC consumes the current chunk;
4. only bounded chunk memory is used, regardless of total video length.

Recommended starting settings:

- `codec = h264_nvenc`
- `preset = p3` for speed, `p4` for a little more compression efficiency
- `nvenc_gpu` defaults to the highest-generation NVIDIA GPU detected through `nvidia-smi` (important on multi-GPU systems)
- `bitrate_mbps = 20` for typical 1080p-ish output; increase for high-resolution/high-detail content
- `pixel_format = yuv420p` for broad compatibility
- `chunk_mb = 64`
- `pipeline_depth = 2`

ProRes export is also available. Select one of the `prores_*` codec profiles; AetherScale automatically writes a `.mov` container and uses the matching `prores_ks` profile. 422 profiles use `yuv422p10le`; 4444/XQ use 10-bit 4:4:4 (and preserve alpha when a 4-channel IMAGE is supplied). The generic `bitrate_mbps`, NVENC GPU, and NVENC preset controls are not used by ProRes.

`audio_bitrate_kbps` is a dropdown with common AAC bitrates from 64 to 512 kbps.

Browser playback of ProRes MOV inside the node depends on the browser/OS codec stack. The encoded MOV remains valid even when the browser cannot decode ProRes for inline playback.

Optional `AUDIO` is written to a temporary PCM WAV with the Python standard library and muxed by FFmpeg as AAC. The temporary audio file is removed after encoding. By default only the final audio-muxed video is saved. Enable `save_silent_copy` only when you explicitly want a second silent copy; it is created with stream-copy and no video re-encode.

The encoded video is shown directly inside the node after execution. Metadata can be controlled independently with `save_metadata` and `metadata_target`:

- `sidecar_json` — writes `<video>.metadata.json` next to the video.
- `video_container` — embeds workflow/prompt metadata into the video container.
- `both` — embeds metadata and writes the JSON sidecar.

Generation metadata is captured automatically from the upstream graph when possible. AetherScale walks back from the IMAGE input and records the nearest relevant seed/noise source, sampler, scheduler, and checkpoint/model loader. The `seed`, `sampler_name`, `scheduler`, and `model_name` optional inputs can be connected when you want to override or supply values explicitly.

For `video_container`, the common generation fields are written as separate human-readable tags (`model`, `seed`, `sampler`, `scheduler`) and also summarized in the container description. This makes it much easier to inspect a file with normal media-info tools without reopening the workflow in ComfyUI. The full workflow/prompt metadata is still preserved in the embedded JSON comment.

When `save_output = false`, the preview file is written to ComfyUI's temp folder and remains viewable in the node.

`AetherScale • MFG Video` remains the fastest route specifically for MFG because it skips creation of the multiplied `IMAGE` batch entirely. Use `AetherScale • Video Combine` when you already have an IMAGE batch and need a fast general-purpose encoder.

## Native DLSS Frame Generation

`AetherScale • MFG` keeps the historical `AetherScaleMFGLab` class ID for workflow compatibility, but v0.8.0+ uses `native_dlssg` as the default implementation.

### Recommended native settings

- `mode = native_dlssg`
- `multiplier = 4x` for the first RTX 50 test
- `native_guide_source = internal_dis`
- `native_auto_bootstrap = true`
- `native_scene_cut_threshold = 0.24`
- connect the real source FPS to `source_frame_rate` when using the IMAGE-output node

The native path does not use the old `artifact_guard`, `emissive_protection`, `thin_detail_protection`, `mv_confidence_threshold`, `fallback_mode`, or `synthesis_mode` controls. Those controls remain visible only because the legacy `surrogate_mv` backend is still available for comparison and old workflows.

### Runtime bootstrap

On first native use AetherScale installs two pinned files from `Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation` commit `2c5b661fb94a236321414300e6269441acb2d13d`:

- `dlssg-worker.exe` — SHA-256 `8a747f9ed613842d5b8b34a811ad43bc1a9466540e2e5a0c8ef4005f0db9e384`, 66,560 bytes
- `nvngx_dlssg.dll` — SHA-256 `135eaf0733c1e37381a8c28abcf7a862404a54132b81787c04e35d09efc5e36f`, 7,519,856 bytes

The files are stored under `ComfyUI-AetherScale/runtime/dlssg_native/`. Downloads use the same direct/proxy/curl/PowerShell resilient transport used by the other AetherScale runtimes, and the hash is verified before execution.

### Native motion guide

The default native guide is deliberately conservative:

```text
current RGBA frame
        ↓
grayscale / ~640 px analysis width
        ↓
OpenCV DIS Medium, current → previous
        ↓
full-resolution FP16 motion field
        ↓
NVIDIA DLSSG
```

This is separate from `AetherScale • Motion Analysis`. `connected_motion` remains available for experiments, but `internal_dis` is the recommended native path because it avoids the torch/glow over-fitting behavior that motivated `mfg_safe` in the legacy surrogate pipeline.

### Multipliers and cascade

The worker reports `multi_frame_count_max` at runtime. If the requested multiplier is natively supported, AetherScale uses one DLSSG history and requests all intermediate frames directly. If not, it creates enough native 2x stages to form a dense temporal grid and selects the nearest frames on the requested exact output timeline. The exposed range is 2x–6x.

For a 24 FPS source, a native 4x run produces frames on the 96 FPS grid. Saving those generated frames at 24 FPS gives true 4x slow motion; saving them at 96 FPS preserves the original duration.

### MFG Video fast path

`AetherScale • MFG Video` is the production path when the next step is a file. Native generated frames are written directly to FFmpeg/NVENC as they arrive from DLSSG, so AetherScale does not allocate a huge 2x/3x/4x/5x/6x IMAGE tensor first.

Use `AetherScale • MFG` instead when you need the generated frames as an IMAGE batch for further ComfyUI processing or frame-by-frame comparison.

### Legacy surrogate

Set `mode = surrogate_mv` (or `backend = surrogate_mv` in MFG Video) only when reproducing pre-v0.8 results or debugging. The legacy path still contains the earlier continuous temporal synthesis, artifact guards, and `mfg_safe` Motion Analysis behavior, but it is no longer the primary MFG implementation.
