from __future__ import annotations

import functools
import sys
import threading
import time
from typing import Any, Callable, TypeVar

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - ComfyUI normally ships tqdm
    tqdm = None  # type: ignore[assignment]

F = TypeVar("F", bound=Callable[..., Any])

_LOCK = threading.RLock()
_ACTIVE: "ConsoleProgress | None" = None


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _comfy_model_management():
    try:
        import comfy.model_management as mm  # type: ignore

        return mm
    except Exception:
        return None


def processing_interrupted() -> bool:
    mm = _comfy_model_management()
    if mm is None:
        return False
    try:
        return bool(mm.processing_interrupted())
    except Exception:
        return False


def throw_if_interrupted() -> None:
    """Honor ComfyUI's global Interrupt button from custom-node loops.

    ComfyUI does not forcibly terminate Python work. Long custom nodes must poll
    the same global interruption flag used by core samplers. Keeping the helper
    here makes every AetherScale backend use one contract and also keeps import
    tests working outside a full ComfyUI installation.
    """
    mm = _comfy_model_management()
    if mm is None:
        return
    mm.throw_exception_if_processing_interrupted()


def is_interrupt_exception(exc: BaseException) -> bool:
    mm = _comfy_model_management()
    if mm is None:
        return False
    cls = getattr(mm, "InterruptProcessingException", None)
    return bool(cls is not None and isinstance(exc, cls))


def _close_active_progress(*, status: str) -> None:
    global _ACTIVE
    with _LOCK:
        active = _ACTIVE
    if active is not None and not active.closed:
        try:
            active.close(status=status)
        except Exception:
            pass


def _status_line(text: str) -> None:
    """Write a normal status line without corrupting an active tqdm bar."""
    with _LOCK:
        if tqdm is not None and _ACTIVE is not None and not _ACTIVE.closed:
            tqdm.write(text, file=sys.stderr)
        else:
            print(text, flush=True)


class ConsoleProgress:
    """One live console + ComfyUI progress bar per long AetherScale stage.

    `tqdm` owns the terminal row, while ComfyUI's own ProgressBar updates the UI
    progress state and, importantly, executes the same interruption check used
    by built-in nodes. The direct `throw_if_interrupted()` poll is retained as a
    fallback and is also used before any work is reported.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "frame",
        min_interval: float = 0.5,
        width: int = 24,  # retained for call-site compatibility
    ) -> None:
        del width
        global _ACTIVE
        self.label = str(label)
        self.total = max(0, int(total))
        self.unit = str(unit)
        self.min_interval = max(0.05, float(min_interval))
        self.current = 0
        self.started = time.perf_counter()
        self.closed = False
        self._last_fallback_bucket = -1
        self._bar = None
        self._comfy_bar = None

        throw_if_interrupted()
        try:
            from comfy.utils import ProgressBar  # type: ignore

            self._comfy_bar = ProgressBar(self.total if self.total > 0 else 1)
        except Exception:
            self._comfy_bar = None

        with _LOCK:
            if _ACTIVE is not None and not _ACTIVE.closed:
                _ACTIVE.close(status="interrupted")
            _ACTIVE = self
            if tqdm is not None:
                self._bar = tqdm(
                    total=self.total if self.total > 0 else None,
                    desc=f"[AetherScale] {self.label}",
                    unit=self.unit,
                    dynamic_ncols=True,
                    mininterval=self.min_interval,
                    maxinterval=max(1.0, self.min_interval * 4.0),
                    smoothing=0.15,
                    leave=True,
                    file=sys.stderr,
                    bar_format=(
                        "{desc} |{bar}| {percentage:6.2f}% | "
                        "{n_fmt}/{total_fmt} {unit}s | {rate_fmt} | ETA {remaining}"
                    ) if self.total > 0 else None,
                )

        if self._bar is None:
            self._fallback_render(force=True)

    def _fallback_line(self) -> str:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        rate = self.current / elapsed if self.current else 0.0
        if self.total > 0:
            pct = min(100.0, max(0.0, self.current * 100.0 / self.total))
            remaining = max(0, self.total - self.current)
            eta = remaining / rate if rate > 1e-9 else 0.0
            return (
                f"[AetherScale] {self.label} {pct:6.2f}% | "
                f"{self.current}/{self.total} {self.unit}s | "
                f"{rate:.2f} {self.unit}/s | ETA "
                f"{_fmt_time(eta) if rate > 1e-9 else '--:--'}"
            )
        return (
            f"[AetherScale] {self.label} | {self.current} {self.unit}s | "
            f"{rate:.2f} {self.unit}/s"
        )

    def _fallback_render(self, *, force: bool = False) -> None:
        if self.closed:
            return
        if self.total > 0:
            bucket = min(10, int(self.current * 10 / max(1, self.total)))
            if not force and bucket == self._last_fallback_bucket:
                return
            self._last_fallback_bucket = bucket
        elif not force:
            return
        print(self._fallback_line(), flush=True)

    def _update_comfy(self, delta: int) -> None:
        throw_if_interrupted()
        if self._comfy_bar is not None and delta > 0:
            # ComfyUI's ProgressBar hook performs another interrupt check and
            # updates the browser-side node progress indicator.
            self._comfy_bar.update(delta)

    def update(self, amount: int = 1) -> None:
        if self.closed:
            return
        amount = int(amount)
        if amount <= 0:
            throw_if_interrupted()
            return
        target = self.current + amount
        if self.total > 0:
            target = min(target, self.total)
        delta = target - self.current
        self._update_comfy(delta)
        self.current = target
        if self._bar is not None:
            self._bar.update(delta)
        else:
            self._fallback_render(force=self.total > 0 and self.current >= self.total)

    def set(self, value: int) -> None:
        if self.closed:
            return
        target = max(0, int(value))
        if self.total > 0:
            target = min(target, self.total)
        delta = target - self.current
        if delta >= 0:
            self._update_comfy(delta)
        else:
            throw_if_interrupted()
        self.current = target
        if self._bar is not None:
            if delta >= 0:
                self._bar.update(delta)
            else:
                self._bar.n = target
                self._bar.refresh()
        else:
            self._fallback_render(force=self.total > 0 and self.current >= self.total)

    def close(self, *, status: str = "done") -> None:
        global _ACTIVE
        if self.closed:
            return
        if status == "done" and self.total > 0 and self.current < self.total:
            delta = self.total - self.current
            self.current = self.total
            # Do not call ComfyUI ProgressBar here: completing a bar during an
            # exception must not mask the original error or re-raise interrupt.
            if self._bar is not None:
                self._bar.update(delta)
        if self._bar is not None:
            self._bar.refresh()
            self._bar.close()
        else:
            self._fallback_render(force=True)
        self.closed = True
        with _LOCK:
            if _ACTIVE is self:
                _ACTIVE = None

    def __enter__(self) -> "ConsoleProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(status="failed" if exc_type is not None else "done")


def console_node(label: str) -> Callable[[F], F]:
    """Wrap a ComfyUI node method with timing and interruption-aware cleanup."""

    def decorate(fn: F) -> F:
        if getattr(fn, "_aetherscale_console_wrapped", False):
            return fn

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any):
            started = time.perf_counter()
            throw_if_interrupted()
            _status_line(f"[AetherScale] START {label}")
            try:
                result = fn(*args, **kwargs)
                throw_if_interrupted()
            except BaseException as exc:
                elapsed = time.perf_counter() - started
                interrupted = is_interrupt_exception(exc) or processing_interrupted()
                _close_active_progress(status="interrupted" if interrupted else "failed")
                verb = "CANCEL" if interrupted else "FAIL "
                _status_line(f"[AetherScale] {verb} {label} after {_fmt_time(elapsed)}")
                raise
            elapsed = time.perf_counter() - started
            _close_active_progress(status="done")
            _status_line(f"[AetherScale] DONE  {label} in {_fmt_time(elapsed)}")
            return result

        setattr(wrapped, "_aetherscale_console_wrapped", True)
        return wrapped  # type: ignore[return-value]

    return decorate
