# AetherScale Native NR Interop Design

AetherScale's `native_interop` Neural Rendering backend is deliberately split
into two layers:

1. **AetherScale-owned Python integration** (`backend/nr_interop.py`)
   - GPU selection/matching for multi-GPU ComfyUI environments
   - bounded per-chunk scheduling
   - scene-reset policy using AetherScale motion packets
   - output precision/storage policy (RAM, anonymous mapping, disk spill)
   - ComfyUI progress and interruption polling
   - one-time channel-order resolution per session/resolution
   - explicit CUDA-interoperability -> CPU-staging fallback

2. **Declared third-party MIT native bridge dependency**
   - project: `orex2121/ComfyUI-DLSS5-orex`
   - pinned commit: `739208e7ae5f576355fc5c30ffb77c4de2e61984`
   - AetherScale downloads the bridge/caller only when this backend is first used
   - downloaded files are checked against their pinned Git blob identities
   - AetherScale does not copy the upstream Python node implementation

This separation is intentional. It gives AetherScale a working in-process
D3D12/NGX/CUDA bridge today while keeping AetherScale's scheduling, storage,
UI contract, fallbacks and long-video behavior independent. The bridge can be
replaced later by an AetherScale-owned native implementation without changing
the ComfyUI node contract or storage pipeline.
