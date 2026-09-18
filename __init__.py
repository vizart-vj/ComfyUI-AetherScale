from __future__ import annotations

try:
    from .backend.storage import startup_cache_cleanup
    _cache_recovery = startup_cache_cleanup()
    _removed = int(_cache_recovery.get("removed_orphans", 0)) + int(_cache_recovery.get("removed_stale", 0))
    if _removed:
        print(
            f"[AetherScale] Startup cache recovery removed {_removed} stale spill file(s). "
            f"Cache: {_cache_recovery.get('cache_dir')}",
            flush=True,
        )
except Exception as _cache_exc:
    # Cache recovery must never prevent ComfyUI from loading the extension/UI.
    print(f"[AetherScale] Startup cache recovery skipped: {_cache_exc}", flush=True)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
