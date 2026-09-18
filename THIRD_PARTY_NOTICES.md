# Third-party notices

AetherScale is an independent project by **noise**. The MIT license in this repository applies only to AetherScale's own source code. Third-party software, runtimes, models, drivers, and binaries retain their original licenses and terms.

## NVIDIA VFX

AetherScale can lazily install the official Python package:

- package: `nvidia-vfx`
- pinned version: `0.1.0.1`
- installation target: AetherScale's private `vendor/` directory

NVIDIA software and trademarks remain subject to NVIDIA's applicable terms.

## DLSS 5 carrier backend

The optional compatibility Neural Rendering carrier backend bootstraps selected runtime files from:

- project: `Merserk/dlss5-visual-enhancer`
- release: `v1.0`
- release asset: `DLSS.5.Visual.Enhancer.v1.0.zip`
- SHA-256: `5d57c2f2d2a1c247c0249e7a1024eabb5384ee9111820a4a478be6ce893b767d`
- upstream: https://github.com/Merserk/dlss5-visual-enhancer

The archive is resolved at runtime only when the carrier backend requires it. AetherScale can use an official local copy or download it through its resilient bootstrap transports, and always verifies the pinned archive hash before extracting the runtime subset. The upstream release contains components with separate licenses/terms, including ReShade/RenoDX/NVIDIA-related files. AetherScale's MIT license does not relicense those components.

## Native in-process DLSS Neural Rendering bridge dependency

AetherScale 0.9.0 introduced an independent `native_interop` adapter in `backend/nr_interop.py`.
The adapter does not copy the upstream Python node implementation. It treats the following
MIT-licensed native files as an explicit third-party dependency and downloads them only
when the backend is first used:

- project: `orex2121/ComfyUI-DLSS5-orex`
- pinned commit: `739208e7ae5f576355fc5c30ffb77c4de2e61984`
- `bridge/bin/dlss5nr_bridge.dll` — pinned Git blob `ffd67747b1272607753743907369e4fb570b0efe`, size 238592 bytes
- `runtime/caller/nvngx.dll_comfy.dll` — pinned Git blob `c69856b68a67a795de27c307919adf2ec7dd0ac2`, size 103424 bytes
- upstream: https://github.com/orex2121/ComfyUI-DLSS5-orex
- verified native NR runtime used by `auto_bootstrap`: `nvngx-v1/nvngx_dlssnr.dll`
- runtime SHA-256: `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`
- runtime size: `165830144` bytes

The NR runtime is NVIDIA software and is not relicensed by AetherScale or by the MIT bridge.
A copy of the upstream MIT license is included as
`third_party/ComfyUI-DLSS5-OreX-LICENSE.txt`. The downloaded bridge/caller remain
third-party software; AetherScale's MIT license does not replace their copyright notice.

AetherScale's own contribution is the surrounding integration architecture: multi-GPU
matching, chunk scheduling, long-video storage/spill, ComfyUI interruption/progress,
scene-reset integration with AetherScale motion packets, channel-order caching, and explicit
CUDA-interoperability fallback. See `native/NR_INTEROP_DESIGN.md`.

## Legacy direct DLSSNR diagnostic backend

The legacy direct backend can bootstrap an MIT bridge/caller from:

- project: `lisitskyaa/ComfyUI-DLSS5-NR`
- release: `v0.2.0`
- SHA-256: `d10d6cd4e7b9d15ef43501baeff1c9fd7b5e3fe41a908b44c338813a82541260`
- upstream: https://github.com/lisitskyaa/ComfyUI-DLSS5-NR

A copy of the upstream MIT license is included in `third_party/ComfyUI-DLSS5-NR-LICENSE.txt`.

The same legacy backend contains optional runtime profiles referencing the public `RankFTW/rhi-repo` release catalog. Current pinned archive hashes in AetherScale 0.9.2 are:

- RTX 50 / `dlssnr-310.8.0`: `388c0a7912e15ec911b9c9e11a692142b11fe387ddf2b637d8c358138fffb3ac`
- RTX 40 / `dlssnr-310.8.0-RTX40`: `46124cfaef532ad5f6da07494772ea8c1b3e719f934e254385697f38d1289e3f`
- fallback / `dlssnr-310.8.SF-v2`: `1da35941894994eb087e017577829e492454e9bae3a6a9397027069ceb74955c`

These downloads are **not** part of the Git repository or Registry package. Users are responsible for ensuring that their use of any third-party runtime is permitted by the applicable license and local law.

## Native DLSS Frame Generation backend

AetherScale v0.9.2 contains a protocol-compatible native DLSS Frame Generation integration based on the public MIT-licensed reference project:

- project: `Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation`
- pinned commit: `2c5b661fb94a236321414300e6269441acb2d13d`
- upstream: https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation

A copy of the reference MIT license is included as `third_party/Konohamaru04-DLSS-Frame-Interpolation-LICENSE.txt`.

The native worker/runtime are not included in the AetherScale source/Registry archives. On first native MFG use, AetherScale may download and verify the exact pinned Git LFS objects:

- `dlssg-worker.exe` — SHA-256 `8a747f9ed613842d5b8b34a811ad43bc1a9466540e2e5a0c8ef4005f0db9e384`
- `nvngx_dlssg.dll` — SHA-256 `135eaf0733c1e37381a8c28abcf7a862404a54132b81787c04e35d09efc5e36f`

`nvngx_dlssg.dll` and other NVIDIA software remain subject to NVIDIA's applicable DLSS license and terms. The NVIDIA license text is fetched beside the runtime when available. AetherScale's MIT license does not relicense NVIDIA binaries or the third-party worker.

## No endorsement

NVIDIA, GeForce RTX, CUDA, DLSS, ReShade, RenoDX, and other product/project names are trademarks or names of their respective owners. Their mention describes compatibility or integration only and does not imply endorsement.


## FFmpeg

AetherScale can invoke an FFmpeg executable already available from the user's environment, PATH, or an installed VideoHelperSuite setup for video encoding. AetherScale does not redistribute FFmpeg in this repository. FFmpeg retains its own applicable license terms and build configuration.
