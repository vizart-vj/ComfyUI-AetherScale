from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

PACKAGE_NAME = "nvidia-vfx"
PACKAGE_VERSION = "0.1.0.1"

_ROOT = Path(__file__).resolve().parents[1]
_VENDOR = _ROOT / "vendor"
_STATE_FILE = _VENDOR / ".aetherscale-runtime.json"
_LOCK = threading.RLock()


@dataclass(slots=True)
class RuntimeState:
    ready: bool
    package: str = PACKAGE_NAME
    requested_version: str = PACKAGE_VERSION
    installed_version: Optional[str] = None
    vendor_path: str = str(_VENDOR)
    python: str = sys.executable
    platform: str = platform.platform()
    message: str = ""


class RuntimeManager:
    """Lazy self-bootstrap for NVIDIA's official nvidia-vfx wheel.

    The package is installed into this custom-node's own vendor/ directory,
    so ComfyUI's global Python environment is not modified.
    """

    @classmethod
    def _activate_vendor(cls) -> None:
        _VENDOR.mkdir(parents=True, exist_ok=True)
        vendor = str(_VENDOR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            # Keep handles alive for process lifetime. NVIDIA's wheel can place
            # dependent DLLs below package-specific nested directories, so discover
            # every directory that actually contains a DLL instead of assuming one
            # fixed wheel layout.
            handles = getattr(cls, "_dll_handles", [])
            registered = getattr(cls, "_dll_registered", set())
            candidates = {_VENDOR}
            if _VENDOR.exists():
                try:
                    candidates.update(p.parent for p in _VENDOR.rglob("*.dll"))
                    candidates.update(p.parent for p in _VENDOR.rglob("*.pyd"))
                except OSError:
                    pass

            for candidate in sorted(candidates, key=lambda p: len(str(p))):
                key = str(candidate.resolve())
                if candidate.is_dir() and key not in registered:
                    try:
                        handles.append(os.add_dll_directory(key))
                        registered.add(key)
                    except OSError:
                        pass
            cls._dll_handles = handles
            cls._dll_registered = registered

    @classmethod
    def _installed_version(cls) -> Optional[str]:
        cls._activate_vendor()
        try:
            from importlib.metadata import version, PackageNotFoundError
            # metadata() searches sys.path distributions, including vendor.
            return version(PACKAGE_NAME)
        except Exception:
            return None

    @classmethod
    def probe(cls) -> RuntimeState:
        cls._activate_vendor()
        version = cls._installed_version()
        try:
            spec = importlib.util.find_spec("nvvfx")
        except Exception:
            spec = None
        ready = spec is not None
        msg = "Runtime ready." if ready else "NVIDIA VFX runtime is not installed yet."
        return RuntimeState(
            ready=ready,
            installed_version=version,
            message=msg,
        )

    @classmethod
    def ensure(cls, *, force_reinstall: bool = False) -> RuntimeState:
        with _LOCK:
            cls._activate_vendor()
            current = cls.probe()
            if current.ready and not force_reinstall:
                return current

            if sys.version_info < (3, 10):
                raise RuntimeError(
                    "AetherScale requires Python 3.10+ because NVIDIA nvidia-vfx "
                    "requires Python 3.10 or newer."
                )

            _VENDOR.mkdir(parents=True, exist_ok=True)
            print(
                f"[AetherScale] Installing NVIDIA VFX runtime {PACKAGE_VERSION} "
                f"into {_VENDOR}. This is automatic and happens only when needed."
            )

            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"{PACKAGE_NAME}=={PACKAGE_VERSION}",
                "--target",
                str(_VENDOR),
                "--no-deps",
                "--disable-pip-version-check",
            ]
            if force_reinstall:
                cmd += ["--upgrade", "--force-reinstall"]
            else:
                cmd += ["--upgrade"]

            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Could not start pip for automatic runtime installation: {exc}"
                ) from exc

            if proc.returncode != 0:
                tail = "\n".join(proc.stdout.splitlines()[-40:])
                raise RuntimeError(
                    "Automatic NVIDIA VFX runtime installation failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"pip output:\n{tail}"
                )

            importlib.invalidate_caches()
            cls._activate_vendor()

            # Remove stale negative/partial import cache entries.
            for name in tuple(sys.modules):
                if name == "nvvfx" or name.startswith("nvvfx."):
                    sys.modules.pop(name, None)

            state = cls.probe()
            if not state.ready:
                raise RuntimeError(
                    "pip completed successfully, but Python still cannot import nvvfx. "
                    "Restart ComfyUI once; if the issue persists, run AetherScale Runtime "
                    "with action=repair."
                )

            try:
                _STATE_FILE.write_text(
                    json.dumps(
                        {
                            **asdict(state),
                            "installed_at": time.time(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

            print(f"[AetherScale] NVIDIA VFX runtime ready ({state.installed_version}).")
            return state

    @classmethod
    def clear(cls) -> RuntimeState:
        with _LOCK:
            # Do not delete loaded native DLLs in-process on Windows. Remove what can
            # be removed and tell the user a restart may be needed.
            for name in tuple(sys.modules):
                if name == "nvvfx" or name.startswith("nvvfx."):
                    sys.modules.pop(name, None)

            if _VENDOR.exists():
                try:
                    shutil.rmtree(_VENDOR)
                except OSError as exc:
                    raise RuntimeError(
                        "Could not fully remove the vendored runtime. Native DLLs may "
                        "still be loaded by this ComfyUI process; restart ComfyUI and "
                        f"run clear again. Original error: {exc}"
                    ) from exc
            _VENDOR.mkdir(parents=True, exist_ok=True)
            return cls.probe()
