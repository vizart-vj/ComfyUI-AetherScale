# GitHub / Comfy Registry publishing checklist

## Repository metadata

Recommended GitHub repository name:

`ComfyUI-AetherScale`

Recommended GitHub description:

> GPU-native NVIDIA video enhancement, restoration, temporal analysis, and experimental DLSS 5 Neural Rendering nodes for ComfyUI.

Recommended topics:

`comfyui`, `nvidia`, `rtx`, `cuda`, `dlss`, `dlss5`, `neural-rendering`, `video-enhancement`, `super-resolution`, `upscaling`, `video-restoration`

## Author / publisher

Public author identity is **noise**.

`pyproject.toml` is prepared with:

```toml
authors = [{ name = "noise" }]

[tool.comfy]
PublisherId = "noise"
DisplayName = "AetherScale"
```

The Comfy Registry publisher ID is globally unique and immutable. Before the first publication, create/verify the Registry publisher `noise` and generate an API key for that publisher.

## Repository URL

The package is prepared for:

`https://github.com/vizart-vj/ComfyUI-AetherScale`

GitHub repository owner and Comfy Registry publisher are intentionally different: GitHub lives under `vizart-vj`, while `PublisherId = "noise"` is used for the Comfy Registry. Keep the project author as `noise`.

## Registry secret

In GitHub:

`Settings -> Secrets and variables -> Actions -> New repository secret`

Create:

`REGISTRY_ACCESS_TOKEN`

and paste the publishing API key created for the `noise` publisher.

## Publishing

The included `.github/workflows/publish-comfy-registry.yml` supports manual publishing and automatically publishes when `pyproject.toml` changes on `main`.

For every release:

1. update `version` in `pyproject.toml`;
2. update the version reported by the nodes/runtime User-Agent;
3. update `README.md` (current version, controls, backend behavior, and release notes);
4. update `CHANGELOG.md` and `THIRD_PARTY_NOTICES.md` when runtime sources/hashes or wording change;
5. verify public author/publisher metadata remains `noise`;
6. commit and push to `main`;
7. verify the `Publish to Comfy Registry` GitHub Action;
8. create the matching GitHub Release/tag.

`ex.png` is intentionally referenced by `README.md` but excluded from the packaged Registry/archive payload. Keep the screenshot in the GitHub repository itself when desired.

The Registry node ID is `aetherscale` and should not be changed after the first successful publication.
