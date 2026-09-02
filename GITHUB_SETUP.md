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

`https://github.com/noise/ComfyUI-AetherScale`

If the actual GitHub repository lives under a different account/organization, update only the URLs under `[project.urls]` before the first Registry publish. Keep `PublisherId = "noise"` if `noise` is the Registry publisher you created.

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
2. update `CHANGELOG.md`;
3. commit and push to `main`;
4. verify the `Publish to Comfy Registry` GitHub Action;
5. create the matching GitHub Release/tag.

The Registry node ID is `aetherscale` and should not be changed after the first successful publication.
