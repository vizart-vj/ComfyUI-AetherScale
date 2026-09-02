
## 0.5.3

- Fixed `AetherScaleHDR` on current NVIDIA VFX runtimes.
- Removed the hard dependency on nonexistent `VideoHDR` / `RTXVideoHDR` Python symbols.
- Added a CUDA-native, frame-streamed HDR-style enhancer fallback using the existing strength, saturation, contrast, highlight-preservation, and profile controls.
- Native NVIDIA VFX HDR remains auto-detectable for future SDK releases.

# Changelog

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
