# AetherScale native integration notes

AetherScale uses small native boundaries for NVIDIA NGX/D3D12 experiments.

## Neural Rendering

The default v0.9.2 Neural Rendering backend is `native_interop`. AetherScale's
Python adapter lives in `backend/nr_interop.py`. It does not copy a third-party
Python node implementation; instead it downloads a pinned MIT native bridge and
caller shim as explicit dependencies on first use. AetherScale provides the
multi-GPU matching, chunk scheduler, output storage, progress/interrupt contract,
scene-reset policy and fallbacks around that bridge.

See `NR_INTEROP_DESIGN.md` and `THIRD_PARTY_NOTICES.md`.

## Legacy host contract

`aetherscale_ngx_host.h` remains as a stable AetherScale-owned boundary for
future native implementations and experiments. The legacy direct feature-18
backend is retained for diagnostics, while the carrier path remains a fallback.
