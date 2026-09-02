# Third-party notices

AetherScale is an independent project by **noise**. The MIT license in this repository applies only to AetherScale's own source code. Third-party software, runtimes, models, drivers, and binaries retain their original licenses and terms.

## NVIDIA VFX

AetherScale can lazily install the official Python package:

- package: `nvidia-vfx`
- pinned version: `0.1.0.1`
- installation target: AetherScale's private `vendor/` directory

NVIDIA software and trademarks remain subject to NVIDIA's applicable terms.

## DLSS 5 carrier backend

The default experimental Neural Rendering carrier backend bootstraps selected runtime files from:

- project: `Merserk/dlss5-visual-enhancer`
- release: `v1.0`
- release asset: `DLSS.5.Visual.Enhancer.v1.0.zip`
- SHA-256: `5d57c2f2d2a1c247c0249e7a1024eabb5384ee9111820a4a478be6ce893b767d`
- upstream: https://github.com/Merserk/dlss5-visual-enhancer

The archive is downloaded at runtime only when the carrier backend requires it. AetherScale verifies the archive hash before extracting the runtime subset. The upstream release contains components with separate licenses/terms, including ReShade/RenoDX/NVIDIA-related files. AetherScale's MIT license does not relicense those components.

## Legacy direct DLSSNR diagnostic backend

The legacy direct backend can bootstrap an MIT bridge/caller from:

- project: `lisitskyaa/ComfyUI-DLSS5-NR`
- release: `v0.2.0`
- SHA-256: `d10d6cd4e7b9d15ef43501baeff1c9fd7b5e3fe41a908b44c338813a82541260`
- upstream: https://github.com/lisitskyaa/ComfyUI-DLSS5-NR

A copy of the upstream MIT license is included in `third_party/ComfyUI-DLSS5-NR-LICENSE.txt`.

The same legacy backend contains optional runtime profiles referencing the public `RankFTW/rhi-repo` release catalog. Current pinned archive hashes in AetherScale 0.5.5 are:

- RTX 50 / `dlssnr-310.8.0`: `388c0a7912e15ec911b9c9e11a692142b11fe387ddf2b637d8c358138fffb3ac`
- RTX 40 / `dlssnr-310.8.0-RTX40`: `46124cfaef532ad5f6da07494772ea8c1b3e719f934e254385697f38d1289e3f`
- fallback / `dlssnr-310.8.SF-v2`: `1da35941894994eb087e017577829e492454e9bae3a6a9397027069ceb74955c`

These downloads are **not** part of the Git repository or Registry package. Users are responsible for ensuring that their use of any third-party runtime is permitted by the applicable license and local law.

## No endorsement

NVIDIA, GeForce RTX, CUDA, DLSS, ReShade, RenoDX, and other product/project names are trademarks or names of their respective owners. Their mention describes compatibility or integration only and does not imply endorsement.
