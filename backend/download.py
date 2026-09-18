from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Iterable, Mapping


class DownloadFailure(RuntimeError):
    """Raised when every supported transport fails to download a file."""


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _stream_urllib(
    url: str,
    destination: Path,
    *,
    user_agent: str,
    timeout: int,
    bypass_proxy: bool,
    headers: Mapping[str, str],
) -> None:
    request_headers = {"User-Agent": user_agent, **dict(headers)}
    req = urllib.request.Request(url, headers=request_headers)
    if bypass_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        response = opener.open(req, timeout=timeout)
    else:
        response = urllib.request.urlopen(req, timeout=timeout)
    with response as resp, destination.open("wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _stream_curl(
    url: str,
    destination: Path,
    *,
    user_agent: str,
    timeout: int,
    headers: Mapping[str, str],
) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise FileNotFoundError("curl executable not found")
    args = [
        curl,
        "--fail",
        "--location",
        "--retry",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        str(max(30, int(timeout))),
        "--user-agent",
        user_agent,
    ]
    for key, value in headers.items():
        args.extend(["--header", f"{key}: {value}"])
    args.extend(["--output", str(destination), url])
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(45, int(timeout) + 30),
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "curl failed").strip()
        raise RuntimeError(f"curl exit {proc.returncode}: {detail[-800:]}")


def _stream_powershell(
    url: str,
    destination: Path,
    *,
    user_agent: str,
    timeout: int,
    headers: Mapping[str, str],
) -> None:
    ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe") or shutil.which("pwsh")
    if not ps:
        raise FileNotFoundError("PowerShell executable not found")
    env = os.environ.copy()
    env["AETHERSCALE_DOWNLOAD_URL"] = url
    env["AETHERSCALE_DOWNLOAD_OUT"] = str(destination)
    env["AETHERSCALE_DOWNLOAD_UA"] = user_agent
    env["AETHERSCALE_DOWNLOAD_ACCEPT"] = str(headers.get("Accept", ""))
    script = (
        "$ErrorActionPreference='Stop';"
        "$ProgressPreference='SilentlyContinue';"
        "$h=@{}; if($env:AETHERSCALE_DOWNLOAD_ACCEPT){$h['Accept']=$env:AETHERSCALE_DOWNLOAD_ACCEPT};"
        "Invoke-WebRequest -UseBasicParsing "
        "-Uri $env:AETHERSCALE_DOWNLOAD_URL "
        "-OutFile $env:AETHERSCALE_DOWNLOAD_OUT "
        "-UserAgent $env:AETHERSCALE_DOWNLOAD_UA -Headers $h"
    )
    proc = subprocess.run(
        [ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(45, int(timeout) + 30),
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "PowerShell download failed").strip()
        raise RuntimeError(f"PowerShell exit {proc.returncode}: {detail[-800:]}")


def _proxy_state() -> str:
    try:
        proxies = urllib.request.getproxies()
    except Exception:
        return "unknown"
    active = sorted(str(k).lower() for k, v in proxies.items() if v)
    return ",".join(active) if active else "none"


def download_file(
    url: str,
    destination: Path,
    *,
    user_agent: str,
    timeout: int = 240,
    extra_urls: Iterable[str] = (),
    headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Download with proxy-bypass and native Windows fallbacks.

    The first attempt intentionally bypasses configured HTTP(S) proxies. A stale
    localhost/system proxy is a common source of WinError 10061 inside desktop
    ComfyUI environments. If direct access is unavailable, the normal proxy-aware
    urllib path, curl, and PowerShell are attempted in turn.
    """
    headers = dict(headers or {})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    _safe_unlink(part)

    urls: list[str] = []
    for candidate in (url, *tuple(extra_urls)):
        candidate = str(candidate or "").strip()
        if candidate and candidate not in urls:
            urls.append(candidate)

    errors: list[str] = []
    transports = (
        ("urllib-direct", lambda u: _stream_urllib(
            u, part, user_agent=user_agent, timeout=timeout, bypass_proxy=True, headers=headers
        )),
        ("urllib-system-proxy", lambda u: _stream_urllib(
            u, part, user_agent=user_agent, timeout=timeout, bypass_proxy=False, headers=headers
        )),
        ("curl", lambda u: _stream_curl(
            u, part, user_agent=user_agent, timeout=timeout, headers=headers
        )),
        ("powershell", lambda u: _stream_powershell(
            u, part, user_agent=user_agent, timeout=timeout, headers=headers
        )),
    )

    # Try the same transport across alternate URLs before moving to a slower
    # fallback. This lets a direct api.github.com asset URL rescue a blocked
    # github.com release URL without first waiting through every proxy/tool path.
    for transport_name, transport in transports:
        for candidate in urls:
            _safe_unlink(part)
            try:
                transport(candidate)
                if not part.is_file() or part.stat().st_size <= 0:
                    raise RuntimeError("download produced an empty file")
                os.replace(part, destination)
                return {
                    "url": candidate,
                    "transport": transport_name,
                    "proxy_state": _proxy_state(),
                }
            except Exception as exc:
                errors.append(
                    f"{transport_name}({candidate}): {type(exc).__name__}: {exc}"
                )

    _safe_unlink(part)
    concise = " | ".join(errors[-8:])
    raise DownloadFailure(
        "All download transports failed. "
        f"Detected proxy configuration: {_proxy_state()}. Attempts: {concise}"
    )


def download_bytes(
    url: str,
    *,
    user_agent: str,
    timeout: int = 180,
    extra_urls: Iterable[str] = (),
    headers: Mapping[str, str] | None = None,
) -> bytes:
    handle = tempfile.NamedTemporaryFile(prefix="aetherscale_download_", delete=False)
    tmp = Path(handle.name)
    handle.close()
    _safe_unlink(tmp)
    try:
        download_file(
            url,
            tmp,
            user_agent=user_agent,
            timeout=timeout,
            extra_urls=extra_urls,
            headers=headers,
        )
        return tmp.read_bytes()
    finally:
        _safe_unlink(tmp)
