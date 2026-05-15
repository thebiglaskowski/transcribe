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
