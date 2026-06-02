import itertools
import logging
import sys
import threading
import time

_BAR_WIDTH = 20
_LABEL_WIDTH = 12


# Internal renderer to encapsulate two-bar cursor state.
# Replaces previous module global `_two_bar_initialized` (which triggered PLW0603
# and was fragile). Public draw_two_bars/reset_two_bars signatures are unchanged
# for minimal caller impact in transcribe.py + tests.
class _TwoBarRenderer:
    def __init__(self) -> None:
        self._initialized: bool = False

    def draw(self, pct: float, eta_str: str, file_index: int, file_total: int) -> None:
        if not _should_render():
            return
        line1 = f"{'Transcribing':<{_LABEL_WIDTH}} {_render_bar(pct)} {pct:3.0f}%  {eta_str}"
        fp = file_index / file_total * 100
        line2 = (
            f"{'Files':<{_LABEL_WIDTH}} {_render_bar(fp)}"
            f" {file_index:>2}/{file_total}  file {file_index} of {file_total}"
        )
        if self._initialized:
            print(f"\033[2A\r{line1}\n\r{line2}", end="", flush=True)
        else:
            print(f"\r{line1}\n\r{line2}", end="", flush=True)
            self._initialized = True

    def reset(self) -> None:
        self._initialized = False
        if _should_render():
            print()


_two_bar_state: dict = {"renderer": None}


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


def draw_two_bars(pct: float, eta_str: str, file_index: int, file_total: int) -> None:
    if _two_bar_state["renderer"] is None:
        _two_bar_state["renderer"] = _TwoBarRenderer()
    _two_bar_state["renderer"].draw(pct, eta_str, file_index, file_total)


def reset_two_bars() -> None:
    if _two_bar_state["renderer"] is not None:
        _two_bar_state["renderer"].reset()
    elif _should_render():
        # Preserve original behavior if never drawn yet
        print()


_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spinner_start(msg: str) -> threading.Thread:
    if not _should_render():
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        return t
    stop = threading.Event()

    def _spin() -> None:
        for frame in itertools.cycle(_FRAMES):
            if stop.is_set():
                break
            print(f"\r{frame} {msg}", end="", flush=True)
            time.sleep(0.08)
        print("\r" + " " * (len(msg) + 3) + "\r", end="", flush=True)

    t = threading.Thread(target=_spin, daemon=True)
    t.stop_event = stop  # type: ignore[attr-defined]
    t.start()
    return t


def spinner_stop(thread: threading.Thread) -> None:
    if hasattr(thread, "stop_event"):
        thread.stop_event.set()
    thread.join(timeout=0.5)
