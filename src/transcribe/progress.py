"""Terminal output: one rich Console owns the screen.

Log lines, finished-item lines and the live progress bars all go through `console`, so
nothing draws over anything else — rich prints each line above the live region and redraws
the bars below it. The live region only runs on a TTY at INFO level; pipes and -q get plain
lines and no bars.
"""

import logging
import unicodedata
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

console = Console(highlight=False)
_logger = logging.getLogger("transcribe")


def live_enabled() -> bool:
    return console.is_terminal and _logger.isEnabledFor(logging.INFO)


def display(text: str | None) -> str:
    """Strip emoji variation selectors and joiners before showing a title.

    Terminals disagree about how wide those sequences are; a one-cell disagreement makes a
    redrawn line wrap, and a wrapped live line smears over the one above it.
    """
    return (text or "").replace("\ufe0f", "").replace("\u200d", "")


def live_title(text: str | None) -> str:
    """Title for the live (redrawn) region: symbol emoji dropped entirely.

    Emoji such as 🛰 or ☀ are drawn one cell wide by some terminals and two by others; in a
    line rich redraws in place, that off-by-one wraps and smears. Printed-once lines keep them.
    """
    kept = "".join(c for c in display(text) if unicodedata.category(c) != "So")
    return " ".join(kept.split())


def fmt_duration(seconds: float | None) -> str:
    """12:31 or 1:18:05."""
    if not seconds:
        return "0:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ConsoleHandler(logging.Handler):
    """Logging → console, so log lines land above the live bars instead of through them.

    A rich Text message is printed as-is (styled lines); anything else is formatted and
    colored by level.
    """

    _STYLES = {logging.WARNING: ("🟡 ", "yellow"), logging.ERROR: ("❌ ", "bold red")}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if isinstance(record.msg, Text):
                console.print(record.msg)
                return
            prefix, style = self._STYLES.get(min(record.levelno, logging.ERROR), ("", None))
            console.print(Text(prefix + self.format(record), style=style))
        except Exception:
            self.handleError(record)


@contextmanager
def status(message: str):
    """Spinner for a one-off wait (model load, channel lookup). Never wrap an input() in it."""
    if not live_enabled():
        yield
        return
    with console.status(Text(message), spinner="dots"):
        yield


class Tracker:
    """The live region of a run: the current item's stage and bar, plus an optional overall bar."""

    def __init__(self, progress: Progress, overall: str | None, total: int) -> None:
        self._progress = progress
        self._item = progress.add_task("", total=None)
        self._overall_name = display(overall)
        self._overall_total = total
        self._done = 0
        self._overall = progress.add_task(self._overall_label(), total=total) if overall else None
        self._title = ""
        self._stage: tuple[str, bool] | None = None

    def _overall_label(self) -> str:
        return f"📺 {self._overall_name}  [{self._done}/{self._overall_total}]"

    def start(self, title: str) -> None:
        """Begin a new item (resets the item bar)."""
        self._title = live_title(title)
        self._stage = None
        self._progress.reset(self._item, total=None, description=f"⏳ {self._title}")

    def stage(self, emoji: str, pct: float | None = None) -> None:
        """Show the item's current step; pct=None shows an indeterminate (pulsing) bar."""
        key = (emoji, pct is None)
        if key != self._stage:  # new step, or working → measurable: restart bar and its ETA
            self._stage = key
            self._progress.reset(
                self._item, total=None if pct is None else 100, description=f"{emoji} {self._title}"
            )
        if pct is not None:
            self._progress.update(self._item, completed=pct)

    def advance(self) -> None:
        """Count the current item as finished (succeeded or failed) on the overall bar."""
        self._done += 1
        if self._overall is not None:
            self._progress.update(self._overall, advance=1, description=self._overall_label())


@contextmanager
def tracker(overall: str | None = None, total: int = 0):
    """Live progress for a run of items. The region vanishes when done, leaving only the
    finished-item lines printed above it."""
    progress = Progress(
        SpinnerColumn(),
        # markup off: titles like "[LIVE] …" must print literally, not as rich style tags
        TextColumn("{task.description}", markup=False, table_column=Column(ratio=1, no_wrap=True)),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TimeRemainingColumn(compact=True),
        console=console,
        transient=True,
        expand=True,  # the description column absorbs the slack and truncates — never wraps
        speed_estimate_period=3600,  # videos take minutes; rich's 30 s default gives no ETA
        disable=not live_enabled(),
    )
    with progress:
        yield Tracker(progress, overall, total)
