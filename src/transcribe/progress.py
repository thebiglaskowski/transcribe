import logging
import sys

_BAR_WIDTH = 20
_LABEL_WIDTH = 12


def _should_render() -> bool:
    return sys.stdout.isatty() and logging.getLogger("transcribe").isEnabledFor(logging.INFO)


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds > 3600:
        return ""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d} remaining"


def _render_bar(pct: float) -> str:
    filled = int(min(pct, 100) / 100 * _BAR_WIDTH)
    return "[" + "█" * filled + "░" * (_BAR_WIDTH - filled) + "]"


def fmt_eta(seconds: float) -> str:
    return _fmt_eta(seconds)


def draw_bar(label: str, pct: float, eta_str: str = "") -> None:
    if not _should_render():
        return
    bar = _render_bar(pct)
    eta = f"  {eta_str}" if eta_str else ""
    print(f"\r{label:<{_LABEL_WIDTH}} {bar} {pct:3.0f}%{eta}", end="", flush=True)


def finish_bar() -> None:
    if not _should_render():
        return
    print()
