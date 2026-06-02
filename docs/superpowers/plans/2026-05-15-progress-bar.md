# Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bare `\rTranscribing 73%` output with an animated ANSI progress bar, ETA, spinner during model loading, download progress for URL mode, and a two-bar batch display — all zero new dependencies.

**Architecture:** A new `progress.py` module owns all ANSI rendering. It exposes six public functions (`draw_bar`, `finish_bar`, `draw_two_bars`, `reset_two_bars`, `spinner_start`, `spinner_stop`) and one public helper (`fmt_eta`). `transcribe.py` and `downloader.py` import from it; no ANSI sequences leak into other modules. All output gates on `sys.stdout.isatty() and logger.isEnabledFor(INFO)` so quiet mode and pipes stay clean.

**Tech Stack:** Python 3.11+, `threading`, `itertools`, `sys`, `time` — stdlib only.

---

## File map

| File | Action | What changes |
|---|---|---|
| `src/transcribe/progress.py` | **Create** | All ANSI rendering logic |
| `src/transcribe/transcribe.py` | **Modify** | Remove `_progress_enabled`; update `transcribe_file`; add spinner to all three `run_*` workflows; pass `file_progress` in batch loops |
| `src/transcribe/downloader.py` | **Modify** | Add `_make_progress_hook`; pass hook + `quiet=True` to yt-dlp |
| `tests/test_unit.py` | **Modify** | Append new tests for `progress.py` |

---

## Task 1: Pure helpers — `_fmt_eta`, `_render_bar`, `_should_render`

**Files:**
- Create: `src/transcribe/progress.py`
- Modify: `tests/test_unit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unit.py`:

```python
# ---------- progress helpers ----------

from transcribe.progress import _fmt_eta, _render_bar


def test_fmt_eta_seconds():
    assert _fmt_eta(30) == "0:30 remaining"


def test_fmt_eta_over_minute():
    assert _fmt_eta(90) == "1:30 remaining"


def test_fmt_eta_zero_returns_empty():
    assert _fmt_eta(0) == ""


def test_fmt_eta_negative_returns_empty():
    assert _fmt_eta(-5) == ""


def test_fmt_eta_over_hour_returns_empty():
    assert _fmt_eta(3700) == ""


def test_render_bar_zero_percent():
    assert _render_bar(0) == "[" + "░" * 20 + "]"


def test_render_bar_full():
    assert _render_bar(100) == "[" + "█" * 20 + "]"


def test_render_bar_half():
    bar = _render_bar(50)
    assert bar == "[" + "█" * 10 + "░" * 10 + "]"


def test_render_bar_clamps_over_100():
    assert _render_bar(150) == "[" + "█" * 20 + "]"
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_unit.py -k "progress" -v
```

Expected: `ModuleNotFoundError: No module named 'transcribe.progress'`

- [ ] **Step 3: Create `src/transcribe/progress.py` with the pure helpers**

(Note: In v0.2.0 the two-bar state was later refactored to a small internal `_TwoBarRenderer` class + state dict to address globals/PLW0603/fragility identified in review. Public API unchanged. See main plan and git history.)

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_unit.py -k "progress" -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/transcribe/progress.py tests/test_unit.py
git commit -m "feat: add progress.py pure helpers (_fmt_eta, _render_bar)"
```

---

## Task 2: `draw_bar`, `finish_bar`, and `fmt_eta` (public)

**Files:**
- Modify: `src/transcribe/progress.py`
- Modify: `tests/test_unit.py`

`fmt_eta` is the public alias for `_fmt_eta` — both `transcribe.py` and `downloader.py` need it. `draw_bar` renders a single overwritten line; `finish_bar` emits the trailing newline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unit.py`:

```python
import sys
import logging
from io import StringIO

from transcribe.progress import draw_bar, finish_bar, fmt_eta


def _tty_buf(monkeypatch) -> StringIO:
    """Return a StringIO that masquerades as a TTY and patch sys.stdout to it."""
    buf = StringIO()
    buf.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", buf)
    logging.getLogger("transcribe").setLevel(logging.INFO)
    return buf


def test_fmt_eta_public_alias():
    assert fmt_eta(30) == "0:30 remaining"


def test_draw_bar_no_output_when_not_tty(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    draw_bar("Test", 50.0, "0:10 remaining")
    assert capsys.readouterr().out == ""


def test_draw_bar_contains_label_pct_eta(monkeypatch):
    buf = _tty_buf(monkeypatch)
    draw_bar("Transcribing", 73.0, "0:12 remaining")
    out = buf.getvalue()
    assert out.startswith("\r")
    assert "Transcribing" in out
    assert "73%" in out
    assert "0:12 remaining" in out


def test_draw_bar_no_eta_when_empty(monkeypatch):
    buf = _tty_buf(monkeypatch)
    draw_bar("Transcribing", 50.0)
    out = buf.getvalue()
    assert "remaining" not in out


def test_finish_bar_emits_newline(monkeypatch):
    buf = _tty_buf(monkeypatch)
    finish_bar()
    assert buf.getvalue() == "\n"


def test_finish_bar_silent_when_not_tty(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    finish_bar()
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_unit.py -k "draw_bar or finish_bar or fmt_eta_public" -v
```

Expected: `ImportError` for `draw_bar`, `finish_bar`, `fmt_eta`.

- [ ] **Step 3: Add `draw_bar`, `finish_bar`, and `fmt_eta` to `progress.py`**

Add after the existing helpers:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_unit.py -k "draw_bar or finish_bar or fmt_eta" -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/transcribe/progress.py tests/test_unit.py
git commit -m "feat: add draw_bar, finish_bar, fmt_eta to progress.py"
```

---

## Task 3: `spinner_start` and `spinner_stop`

**Files:**
- Modify: `src/transcribe/progress.py`
- Modify: `tests/test_unit.py`

The spinner runs on a daemon thread and cycles braille frames every 80ms. When rendering is disabled (non-TTY or quiet), `spinner_start` returns a lightweight do-nothing thread so callers never need to branch.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unit.py`:

```python
import time

from transcribe.progress import spinner_start, spinner_stop


def test_spinner_stop_noop_thread_does_not_raise():
    # In test environment stdout is not a TTY, so spinner_start returns a no-op thread.
    thread = spinner_start("Loading...")
    spinner_stop(thread)
    assert not thread.is_alive()


def test_spinner_stop_live_thread_joins_cleanly(monkeypatch):
    buf = StringIO()
    buf.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", buf)
    logging.getLogger("transcribe").setLevel(logging.INFO)

    thread = spinner_start("Loading model...")
    time.sleep(0.05)
    spinner_stop(thread)

    assert not thread.is_alive()
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_unit.py -k "spinner" -v
```

Expected: `ImportError` for `spinner_start`, `spinner_stop`.

- [ ] **Step 3: Add spinner to `progress.py`**

Add at the top of the file (after the existing imports):

```python
import itertools
import threading
import time
```

Add after `finish_bar`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_unit.py -k "spinner" -v
```

Expected: both tests pass. (The live-thread test may take ~50ms.)

- [ ] **Step 5: Run full suite**

```bash
pytest -v
```

Expected: all 26 original tests + new spinner tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/transcribe/progress.py tests/test_unit.py
git commit -m "feat: add spinner_start / spinner_stop to progress.py"
```

---

## Task 4: `draw_two_bars` and `reset_two_bars`

**Files:**
- Modify: `src/transcribe/progress.py`
- Modify: `tests/test_unit.py`

Two-bar rendering for batch mode. First render prints both lines; subsequent renders emit `\033[2A` (cursor up 2) and reprint. `reset_two_bars` clears the initialized flag and emits a final newline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unit.py`:

```python
from transcribe.progress import draw_two_bars, reset_two_bars


def test_draw_two_bars_first_render_no_cursor_up(monkeypatch):
    buf = _tty_buf(monkeypatch)
    reset_two_bars()  # ensure clean state
    draw_two_bars(50.0, "0:05 remaining", 1, 3)
    out = buf.getvalue()
    assert "\033[2A" not in out
    assert "Transcribing" in out
    assert "Files" in out
    reset_two_bars()


def test_draw_two_bars_second_render_has_cursor_up(monkeypatch):
    buf = _tty_buf(monkeypatch)
    reset_two_bars()
    draw_two_bars(40.0, "0:06 remaining", 1, 3)
    buf.seek(0); buf.truncate()
    draw_two_bars(50.0, "0:05 remaining", 1, 3)
    assert "\033[2A" in buf.getvalue()
    reset_two_bars()


def test_reset_two_bars_clears_initialized_flag(monkeypatch):
    buf = _tty_buf(monkeypatch)
    reset_two_bars()
    draw_two_bars(50.0, "0:05 remaining", 1, 3)  # sets initialized=True
    reset_two_bars()
    buf.seek(0); buf.truncate()
    draw_two_bars(50.0, "0:05 remaining", 1, 3)  # should NOT have cursor-up
    assert "\033[2A" not in buf.getvalue()
    reset_two_bars()
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_unit.py -k "two_bars" -v
```

Expected: `ImportError` for `draw_two_bars`, `reset_two_bars`.

- [ ] **Step 3: Add two-bar functions to `progress.py`**

Add a module-level flag after the imports block (near the top):

```python
_two_bar_initialized: bool = False
```

Add after `finish_bar`:

```python
def draw_two_bars(pct: float, eta_str: str, file_index: int, file_total: int) -> None:
    global _two_bar_initialized
    if not _should_render():
        return
    line1 = f"{'Transcribing':<{_LABEL_WIDTH}} {_render_bar(pct)} {pct:3.0f}%  {eta_str}"
    fp = file_index / file_total * 100
    line2 = f"{'Files':<{_LABEL_WIDTH}} {_render_bar(fp)} {file_index:>2}/{file_total}  file {file_index} of {file_total}"
    if _two_bar_initialized:
        print(f"\033[2A\r{line1}\n\r{line2}", end="", flush=True)
    else:
        print(f"\r{line1}\n\r{line2}", end="", flush=True)
        _two_bar_initialized = True


def reset_two_bars() -> None:
    global _two_bar_initialized
    _two_bar_initialized = False
    if _should_render():
        print()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_unit.py -k "two_bars" -v
```

Expected: all three pass.

- [ ] **Step 5: Run full suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/transcribe/progress.py tests/test_unit.py
git commit -m "feat: add draw_two_bars / reset_two_bars to progress.py"
```

---

## Task 5: Update `transcribe_file` and batch workflows in `transcribe.py`

**Files:**
- Modify: `src/transcribe/transcribe.py`

Three changes in one file:
1. Remove `_progress_enabled()` and its usage; import from `progress`.
2. Update `transcribe_file`: replace progress block, add `file_progress` param.
3. Wrap `WhisperModel(...)` with spinner in all three workflow functions.
4. Pass `file_progress=(i, len(...))` in the two batch loops.

- [ ] **Step 1: Update imports at the top of `transcribe.py`**

Replace the current import block with:

```python
import argparse
import logging
import time
from pathlib import Path

from faster_whisper import WhisperModel

from .downloader import download_audio
from .output import write_srt, write_txt
from .progress import (
    draw_bar,
    draw_two_bars,
    finish_bar,
    fmt_eta,
    reset_two_bars,
    spinner_start,
    spinner_stop,
)
from .prompts import prompt_for_audio_file, prompt_project_name
from .utils import SUPPORTED_EXTENSIONS, default_compute_type, default_device

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Remove `_progress_enabled` and update `transcribe_file` signature and progress block**

Delete the function (lines 16–17):

```python
def _progress_enabled() -> bool:
    return logger.isEnabledFor(logging.INFO)
```

Change the `transcribe_file` signature from:

```python
def transcribe_file(
    audio_path: Path,
    txt_path: Path,
    srt_path: Path | None,
    model: WhisperModel,
    args: argparse.Namespace,
    info_prefix: str = "",
) -> bool:
```

to:

```python
def transcribe_file(
    audio_path: Path,
    txt_path: Path,
    srt_path: Path | None,
    model: WhisperModel,
    args: argparse.Namespace,
    info_prefix: str = "",
    file_progress: tuple[int, int] | None = None,
) -> bool:
```

Replace the entire progress block (currently lines 44–58, the `segments: list = []` through `if show_progress: print()`) with:

```python
    segments: list = []
    for segment in segments_iter:
        segments.append(segment)
        if info.duration:
            pct = min(100.0, segment.end / info.duration * 100)
            elapsed = time.time() - start_time
            remaining = elapsed / max(pct / 100, 0.001) - elapsed
            eta = fmt_eta(remaining)
        else:
            pct = 0.0
            eta = ""
        if file_progress is not None:
            draw_two_bars(pct, eta, *file_progress)
        else:
            draw_bar("Transcribing", pct, eta)

    if file_progress is not None:
        reset_two_bars()
    else:
        finish_bar()
```

- [ ] **Step 3: Add spinner around model loading in `run_project_workflow`**

Find this block in `run_project_workflow`:

```python
    logger.info("\nLoading model: %s", args.model)
    logger.info("Device: %s | Compute type: %s", device, compute_type)
    logger.info("Project folder: %s", project_dir)
    logger.info("Videos to process: %d", len(urls))

    model = WhisperModel(args.model, device=device, compute_type=compute_type)
```

Replace with:

```python
    logger.info("\nProject folder: %s", project_dir)
    logger.info("Videos to process: %d", len(urls))

    _spinner = spinner_start(f"Loading model {args.model}…")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)
```

- [ ] **Step 4: Add spinner around model loading in `run_multi_file_workflow`**

Find:

```python
    logger.info("\nLoading model: %s", args.model)
    logger.info("Device: %s | Compute type: %s", device, compute_type)
    logger.info("Files to process: %d", len(paths))

    model = WhisperModel(args.model, device=device, compute_type=compute_type)
```

Replace with:

```python
    logger.info("\nFiles to process: %d", len(paths))

    _spinner = spinner_start(f"Loading model {args.model}…")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)
```

- [ ] **Step 5: Add spinner around model loading in `run_single_file_workflow`**

Find:

```python
    logger.info("")
    logger.info("Loading model: %s", args.model)
    logger.info("Device: %s | Compute type: %s", device, compute_type)

    model = WhisperModel(args.model, device=device, compute_type=compute_type)
```

Replace with:

```python
    logger.info("")
    _spinner = spinner_start(f"Loading model {args.model}…")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)
```

- [ ] **Step 6: Pass `file_progress` in the `run_project_workflow` loop**

Find in the `run_project_workflow` for-loop:

```python
            ok = transcribe_file(audio_path, txt_path, srt_path, model, args, info_prefix="  ")
```

Replace with:

```python
            ok = transcribe_file(
                audio_path, txt_path, srt_path, model, args,
                info_prefix="  ", file_progress=(i, len(urls)),
            )
```

- [ ] **Step 7: Pass `file_progress` in the `run_multi_file_workflow` loop**

Find in the `run_multi_file_workflow` for-loop:

```python
            ok = transcribe_file(audio_path, txt_path, srt_path, model, args, info_prefix="  ")
```

Replace with:

```python
            ok = transcribe_file(
                audio_path, txt_path, srt_path, model, args,
                info_prefix="  ", file_progress=(i, len(paths)),
            )
```

- [ ] **Step 8: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass. (No new tests for the workflow functions — they require `WhisperModel` and are out of unit-test scope per project convention.)

- [ ] **Step 9: Commit**

```bash
git add src/transcribe/transcribe.py
git commit -m "feat: integrate progress bars and model-loading spinner into workflows"
```

---

## Task 6: Add download progress hook to `downloader.py`

**Files:**
- Modify: `src/transcribe/downloader.py`
- Modify: `tests/test_unit.py`

yt-dlp accepts a `progress_hooks` list. We provide a closure that calls `draw_bar` on `"downloading"` events and `finish_bar` on `"finished"`. Setting `"quiet": True` suppresses yt-dlp's own output so only our bar is visible.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unit.py`:

```python
from transcribe.downloader import _make_progress_hook


def test_progress_hook_downloading_calls_draw_bar(monkeypatch):
    drawn = []
    # Patch the name as imported in downloader.py, not in progress.py
    monkeypatch.setattr("transcribe.downloader.draw_bar", lambda *a, **kw: drawn.append(a))
    hook = _make_progress_hook()
    hook({
        "status": "downloading",
        "downloaded_bytes": 50_000_000,
        "total_bytes": 100_000_000,
    })
    assert len(drawn) == 1
    label, pct = drawn[0][0], drawn[0][1]
    assert label == "Downloading"
    assert abs(pct - 50.0) < 0.1


def test_progress_hook_finished_calls_finish_bar(monkeypatch):
    finished = []
    monkeypatch.setattr("transcribe.downloader.finish_bar", lambda: finished.append(True))
    hook = _make_progress_hook()
    hook({"status": "finished"})
    assert finished == [True]


def test_progress_hook_skips_when_total_unknown(monkeypatch):
    drawn = []
    monkeypatch.setattr("transcribe.downloader.draw_bar", lambda *a, **kw: drawn.append(a))
    hook = _make_progress_hook()
    hook({"status": "downloading", "downloaded_bytes": 1000})
    assert drawn == []
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_unit.py -k "progress_hook" -v
```

Expected: `ImportError` for `_make_progress_hook`.

- [ ] **Step 3: Add `_make_progress_hook` to `downloader.py`**

Add these imports at the top of `downloader.py`:

```python
import time

from .progress import draw_bar, finish_bar, fmt_eta
```

Add this function before `download_audio`:

```python
def _make_progress_hook():
    start = [time.time()]

    def hook(d: dict) -> None:
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if not total:
                return
            pct = d["downloaded_bytes"] / total * 100
            elapsed = time.time() - start[0]
            remaining = elapsed / max(pct / 100, 0.001) - elapsed
            draw_bar("Downloading", pct, fmt_eta(remaining))
        elif d["status"] == "finished":
            finish_bar()

    return hook
```

- [ ] **Step 4: Wire the hook into `download_audio`**

In the `ydl_opts` dict, change `"quiet": False` to `"quiet": True` and add `"progress_hooks"`:

```python
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=opus]/bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "keepvideo": False,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_make_progress_hook()],
    }
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_unit.py -k "progress_hook" -v
```

Expected: all three pass.

- [ ] **Step 6: Run full suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Lint**

```bash
ruff check src/ tests/ && ruff format src/ tests/
```

Fix any issues, then:

- [ ] **Step 8: Commit**

```bash
git add src/transcribe/downloader.py tests/test_unit.py
git commit -m "feat: add download progress bar via yt-dlp progress hook"
```

---

## Done

All six public `progress.py` functions are wired up. Verify manually with a real audio file:

```bash
# Single file — should see spinner then transcription bar
transcribe path/to/audio.mp3

# Batch — should see spinner, then two bars per file
transcribe file1.mp3 file2.mp3

# URL mode — should see spinner, download bar, then transcription bars
transcribe https://youtu.be/...

# Quiet mode — no progress output at all
transcribe -q path/to/audio.mp3

# Piped — no ANSI noise
transcribe path/to/audio.mp3 2>&1 | cat
```
