# Changelog

## 0.9.2

- Fixed `AetherScale • Video Combine` progressively growing taller after executions and ordinary UI interactions.
- Reworked the player layout around a single persistent LiteGraph DOM widget whose `computeSize()` reports an absolute aspect-ratio-derived height.
- Removed the old `baseHeight + previewHeight` resize path; the previous implementation captured a height that already included the preview and added the preview again on the next refresh.
- Re-encoding now updates the existing player instead of deleting and re-adding the preview widget.
- Width changes re-fit the preview from the video aspect ratio; vertical resizing no longer feeds back into another height increase.
- Added automatic layout healing for workflows saved with the runaway Combine height from older builds.
- No VHS frontend code is bundled or copied; VHS Video Combine was used only as a behavioral/layout reference while the AetherScale widget was independently implemented.
- Synchronized package/runtime version metadata and documentation for v0.9.2.

## 0.9.1

- Added `AetherScale • Video Loader` for low-RAM FFmpeg video input.
- Replaced whole-video float32 materialization with chunked RGB24 decode → in-place FP16/FP32 normalization → sequential backing-file writes.
- `precision=auto` selects FP16 for large sources; a 2464×512, 15 s, 24 fps RGB sequence drops from about 5.08 GiB float32 storage to about 2.54 GiB FP16 backing storage, with bounded working RAM.
- Added disk-space preflight, bounded 128 MiB sequential writeback, ComfyUI interruption, decode progress, audio extraction, source-FPS preservation, `force_rate`, `start_time`, and `frame_load_cap`.
- Added reusable storage helpers for mapping a fully-written sequential tensor file without zero-filling or dirtying the whole mapping first.
- Added Video Loader to Diagnostics and synchronized package version, User-Agent strings, README, third-party notices, and release metadata for v0.9.1.

## 0.9.0

- Neural Rendering now defaults to the new `native_interop` backend.
- Added AetherScale-owned in-process NR adapter with CUDA/D3D12 device-pointer handoff when the bridge supports it.
- Added pinned, checksum-verified first-use bootstrap of the MIT OreX native bridge/caller as explicit third-party dependencies; no OreX Python node implementation is copied into AetherScale.
- Added multi-GPU adapter matching by native bridge GPU name instead of assuming CUDA and DXGI ordinals are identical.
- Added bounded native NR chunk processing with automatic per-chunk VRAM sizing and persistent temporal bridge history.
- Preserved AetherScale FP16/RAM/anonymous-map/disk-spill output storage instead of forcing a full float32 long-video result tensor.
- Added direct GPU result mode for safely sized GPU-resident IMAGE inputs.
- Added explicit native NR fallback policy: `carrier`, `legacy_direct`, or `error`.
- Added `native_chunk_frames` and `native_scene_change_threshold` optional controls.
- CUDA-interoperability failures now fall back to CPU staging inside the same in-process bridge session and report the exact reason.
- Channel-order auto-detection is cached per bridge session/resolution instead of forcing per-frame GPU synchronization.
- Existing required input contract for `AetherScaleNeuralRendering` remains frozen and unchanged.
- Added `native/NR_INTEROP_DESIGN.md` and the upstream OreX MIT license to make the integration boundary and attribution explicit.
- Synchronized README, Registry/package metadata, User-Agent strings, and third-party notices for v0.9.0.

## 0.8.7

- Added native ComfyUI interruption polling to all long AetherScale progress stages and queue waits.
- AetherScale progress now also updates ComfyUI `ProgressBar`, connecting browser progress and Interrupt behavior.
- Reworked NVIDIA VFX Super Resolution into a two-frame GPU/CPU overlapped pipeline so disk-spill writes no longer serialize the GPU stage.
- Removed whole-process Windows working-set trimming introduced in v0.8.6; it could evict unrelated model/tensor pages and create severe CPU/page-fault pressure.
- Reduced default bounded spill flush interval from 256 MiB to 128 MiB while keeping writeback controlled.
- Amortized aggressive CUDA cache cleanup to avoid synchronizing the allocator on every VSR frame.
- Video Combine queue waits and MFG Video progress are now interruption-aware, and duplicate ComfyUI progress emitters were removed.

## 0.8.6

- Fixed HDR large-batch CUDA OOM caused by whole-batch pinned float32 CPU allocation.
- HDR now uses low-memory storage with automatic FP16 and mmap/disk spill.
- Added optional HDR output precision/storage/cache controls.
- Native MFG IMAGE output now fails fast above the configurable safety threshold instead of attempting 100+ GiB allocations; direct users to MFG Video streaming.
- Synchronized documentation and release metadata.

## 0.8.4

- Replaced the custom carriage-return console progress renderer with ComfyUI-style `tqdm` progress bars.
- Added dynamic terminal-width handling so progress details do not wrap and accidentally create repeated rows.
- Added sparse 10% milestone fallback when `tqdm` is unavailable.
- Preserved START/DONE/FAIL timing and all existing stage progress instrumentation.
- Synchronized release documentation and metadata for v0.8.4.

## 0.8.3

- Added import-time cleanup of dead-process disk-spill mmap files from both the current and legacy AetherScale cache directories.
- Moved the default large spill location to ComfyUI's temp directory when available.
- Added `RECOVER_AETHERSCALE_CACHE.bat` for emergency cache cleanup while ComfyUI is closed.
- Hardened `web/video_preview.js` so optional preview failures cannot prevent the ComfyUI frontend from loading.
- Replaced recursive node `computeSize()` preview fitting with deterministic base-height + aspect-ratio sizing.
- Synchronized release documentation and metadata for v0.8.3.

## 0.8.2

- Reworked console progress into a true in-place single-line bar using carriage-return updates instead of one `print()` per refresh.
- Added optional `nvenc_codec_fallback` to `AetherScale • Video Combine`, enabled by default.
- When the requested NVENC codec cannot open at the real output geometry, Video Combine now probes compatible NVENC codecs and continues with the first working one instead of failing after upstream processing.
- Added requested/resolved codec and codec-fallback telemetry to Video Combine stats.
- Fixed a duplicate progress increment in native `AetherScale • MFG Video`.
- Synchronized release documentation and metadata for v0.8.2.

## 0.8.1

- Fixed false NVENC preflight failures on current NVIDIA drivers caused by probing H.264 NVENC with an invalid 64×64 test frame. Preflight now uses the actual output geometry and requested pixel format.
- Applied the same NVENC preflight fix to `AetherScale • MFG Video`.
- Added dependency-free console progress reporting to AetherScale long-running frame pipelines with percent, frame/pair count, throughput, ETA, and elapsed/average speed.
- Every AetherScale node now prints start, finish, and failure timing lines in the console.
- Added detailed progress reporting for Super Resolution, Restoration, HDR, Motion Analysis, Neural Rendering, native/legacy MFG, MFG Video, and Video Combine.
- Synchronized release documentation and metadata for v0.8.1.

## 0.8.0

- Replaced the default MFG surrogate with a native D3D12 NVIDIA DLSS Frame Generation backend.
- Added verified lazy bootstrap for the pinned `dlssg-worker.exe` and `nvngx_dlssg.dll` from `Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation` commit `2c5b661fb94a236321414300e6269441acb2d13d`.
- Added direct RGBA8 + FP16 motion binary protocol and native generated-frame readback.
- Added OpenCV DIS Medium current→previous guide generation as the recommended native MFG motion path.
- Added runtime capability probing via `--probe`, including `multi_frame_count_max`.
- Added native exact multiplier planning and automatic 2x cascade fallback for requested 2x–6x output grids.
- Upgraded `AetherScale • MFG Video` to stream native generated frames directly into FFmpeg/NVENC.
- Preserved the legacy `surrogate_mv` implementation for workflow compatibility and diagnostics.
- Preserved the frozen required schema of `AetherScaleNeuralRendering`.
- Synchronized release documentation and metadata for v0.8.0.

## 0.7.9

- Fixed NVENC `No capable devices found` on multi-GPU Windows systems caused by passing a physical `nvidia-smi` ordinal directly to FFmpeg's logical CUDA `-gpu` option.
- Physical GPU selection now remaps the FFmpeg child process with `CUDA_VISIBLE_DEVICES` and uses logical GPU 0.
- Added a tiny NVENC encoder preflight before real video streaming.
- Added automatic probe fallback to process-visible NVENC when a driver/FFmpeg build rejects explicit remapping.
- Added NVENC routing/preflight telemetry to Video Combine stats.
- Synchronized release documentation and metadata for v0.7.9.

## 0.7.8

- Fixed `WinError 1450` from huge anonymous/pagefile-backed output mappings on Windows.
- Added Windows commit preflight for large mapped outputs.
- Added automatic temporary disk-backed spill mmap fallback when anonymous mapping cannot be reserved.
- Added disk-space preflight and actionable failure messages when spill capacity is insufficient.
- Added `AETHERSCALE_CACHE_DIR` override for placing large spill mappings on another drive.
- Added storage fallback/commit/disk telemetry to output stats.
- Preserved PID-aware orphan cleanup and automatic deletion of clean-cache spill files when tensors are released.
- Synchronized release documentation and metadata for v0.7.8.

## 0.7.7

- Added `mfg_safe` to `AetherScale • Motion Analysis` and made it the default for new MFG-oriented motion analysis.
- Changed MFG internal motion generation from the old `balanced` preset to `mfg_safe`.
- Reworked `balanced` and `quality` as robust refinements anchored to the stable `fast` flow instead of independent LK solutions.
- Added photometric-improvement gating, flow-disagreement gating, and brightness-volatility confidence reduction.
- Prevents higher LK settings from over-fitting torch flicker, glow, sparks, and generative micro-detail into false motion vectors.
- Preserved existing Motion Analysis input names and existing Neural Rendering frozen schema.
- Synchronized release documentation and metadata for v0.7.7.

## 0.7.4

- Added automatic upstream generation metadata extraction in `AetherScale • Video Combine`.
- Captures seed, sampler, scheduler, model/checkpoint, steps, CFG, and denoise when present in the connected graph.
- Added optional connectable overrides: `seed`, `sampler_name`, `scheduler`, and `model_name`.
- Embedded video metadata now exposes human-readable model/seed/sampler/scheduler tags in addition to the full workflow JSON.
- Sidecar JSON now contains a structured `generation` block.
- Preserved existing Video Combine required inputs and Neural Rendering frozen schema.
- Synchronized release documentation and metadata for v0.7.4.

## 0.7.3

- Changed `audio_bitrate_kbps` in `AetherScale • Video Combine` to a dropdown with common AAC bitrate presets.
- Added MOV container output.
- Added Apple ProRes profiles: Proxy, LT, Standard, HQ, 4444, and 4444 XQ through FFmpeg `prores_ks`.
- ProRes automatically resolves to MOV and 10-bit output pixel formats.
- ProRes input packing uses RGB48/RGBA64 to avoid an unnecessary RGB8 precision bottleneck before encoding.
- Preserved inline preview, metadata, audio mux, and silent-copy controls.
- Synchronized release documentation and metadata for v0.7.3.

## 0.7.1

- Added inline video preview to `AetherScale • Video Combine`.
- Added `save_metadata` and `metadata_target` (`sidecar_json`, `video_container`, `both`).
- Added prompt/workflow metadata embedding and `.metadata.json` sidecar export.
- Added `save_silent_copy`; audio-connected workflows now save one audio-muxed video by default and only create a silent duplicate when explicitly requested.
- Temp outputs now use ComfyUI's temp directory for reliable in-node preview.
- Added AetherScale frontend preview widget with no VHS dependency.
- Synchronized release documentation and metadata for v0.7.1.

## 0.7.0

- Added `AetherScale • Video Combine`, a generic high-throughput IMAGE-to-video output node.
- Replaced per-frame NumPy/bytes conversion with chunked Torch RGB8 packing and direct buffer-protocol writes.
- Added bounded producer/consumer pipelining to overlap frame packing with FFmpeg/NVENC consumption.
- Added NVENC H.264/HEVC/AV1 codecs with libx264 fallback.
- Added optional ComfyUI AUDIO muxing to AAC.
- Added `chunk_mb` and `pipeline_depth` performance controls.
- Synchronized release documentation and metadata for v0.7.0.

## 0.6.3

- Added artifact guards to `AetherScale • MFG Lab` and `AetherScale • MFG Video`.
- New controls: `artifact_guard`, `emissive_protection`, `thin_detail_protection`, `mv_confidence_threshold`, and `fallback_mode`.
- Guard path now uses motion confidence, warp residuals, high-frequency detail masks, and emissive-risk masks to suppress doubled thin bright details.
- Safer fallback reconstruction reduces ghosting on torches, sparks, and small glowing accents.
- Synchronized release documentation and metadata for v0.6.3.



## 0.6.2

- Added `AetherScale • MFG Video` direct streaming encoder output.
- MFG frames are generated and written immediately to FFmpeg/NVENC; no full multiplied IMAGE batch is allocated or reread during export.
- Added automatic output FPS calculation: `source_frame_rate × multiplier`.
- Added H.264/HEVC/AV1 NVENC choices and `libx264` compatibility fallback.
- Added direct packed RGB24 conversion that avoids VHS-style float NumPy frame intermediates.
- Preserved `AetherScale • MFG Lab` for workflows that require IMAGE output.
- Synchronized release documentation and metadata for v0.6.2.

## 0.6.1

- Fixed MFG Lab long-video `DefaultCPUAllocator` failures caused by final `torch.stack(outputs)` materialization.
- Replaced list accumulation/final stack with a single preallocated streaming output writer.
- Added automatic FP16 for large MFG outputs.
- Added `output_precision`, `output_storage`, and `clean_cache` controls to MFG Lab.
- Large clean-cache outputs can use anonymous/pagefile-backed mmap with no persistent cache file.
- Prevented large completed MFG batches from being copied wholesale back to CUDA.
- Synchronized release documentation and metadata for v0.6.1.


## 0.6.0

- Added `AetherScale • MFG Lab`.
- Added a machine readiness probe for likely Streamline DLSS-G / MFG support.
- Added a motion-vector-guided surrogate frame generator to study vector-driven interpolation versus RGB-only methods such as RIFE.
- MFG Lab can use connected `AetherScale • Motion Analysis` packets or generate compact internal motion automatically.
- Fully synchronized release metadata and documentation for v0.6.0.

## 0.5.6

- Fixed DLSS5 carrier first-run bootstrap failures caused by rejecting/stale Windows proxy configuration (`WinError 10061`).
- Added proxy-bypass urllib, system-proxy urllib, curl, and PowerShell download transports with atomic partial-file handling and a public GitHub API asset fallback.
- Added automatic local discovery of the pinned Visual Enhancer v1.0 archive in the Windows Downloads folder.
- Added optional `AETHERSCALE_CARRIER_ARCHIVE` and `AETHERSCALE_CARRIER_URL` overrides while preserving strict SHA-256 verification.
- Reused the resilient downloader for legacy DLSSNR runtime/bridge bootstrap.
- Preserved v0.5.5 anonymous/pagefile-backed clean-cache behavior and stable serialized node contracts.
- Updated README, package/Registry metadata, User-Agent strings, and release documentation.

## 0.5.5

- Fixed `clean_cache=true` semantics on Windows/ComfyUI execution caching.
- Replaced clean-cache disk-backed outputs with anonymous/pagefile-backed mappings so completed workflows create no persistent `.mmap` files.
- Kept `clean_cache=false` as explicit file-backed mmap mode.
- Preserved PID-aware orphan cleanup for legacy/persistent mmap files.
- Updated README, package/Registry metadata, User-Agent strings, and release documentation.

## 0.5.4

- Added `clean_cache` to mmap-backed Super Resolution and Neural Rendering outputs.
- Automatically removes mmap cache files when ComfyUI releases their tensor storage.
- Cleans orphaned files from dead processes while protecting live mappings.
- Removed the old process-lifetime memmap keepalive that caused large cache files to accumulate.
- Synchronized README, Registry/package metadata, runtime User-Agent, author/publisher identity, and third-party notices for the v0.5.4 release.


## 0.5.3

- Fixed `AetherScaleHDR` on current NVIDIA VFX runtimes.
- Removed the hard dependency on nonexistent `VideoHDR` / `RTXVideoHDR` Python symbols.
- Added a CUDA-native, frame-streamed HDR-style enhancer fallback using the existing strength, saturation, contrast, highlight-preservation, and profile controls.
- Native NVIDIA VFX HDR remains auto-detectable for future SDK releases.

## 0.5.2

- fixed carrier GPU-ranking runtime regression (`re` import)
- added Windows high-performance GPU routing for the D3D12 carrier worker
- carrier backend remains the default Neural Rendering architecture
- current-to-previous temporal motion flow with compact FP16 storage
- long-video mmap/FP16 memory safeguards retained

## 0.5.0

- replaced the default naked feature-18 path with a DLSS carrier architecture
- added 1x / 1.5x / 1.724x / 2x / 3x DLSS output modes
- added internal DIS temporal motion fallback
- retained legacy direct backend for diagnostics only

## 0.4.x

- introduced experimental DLSS 5 Neural Rendering support
- added Motion Analysis and VRAM planning
- added streaming/mmap long-video memory architecture

## 0.2.x – 0.3.x

- expanded NVIDIA VFX Super Resolution into restoration and HDR-oriented nodes
- established stable serialized node contracts and low-VRAM processing

## 0.1.x

- initial NVIDIA VFX Super Resolution node and private lazy runtime bootstrap
