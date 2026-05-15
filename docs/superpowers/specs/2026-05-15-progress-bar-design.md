# Progress Bar Design

**Date:** 2026-05-15
**Status:** Approved

## Summary

Add animated, zero-dependency ANSI progress output to the transcribe CLI across three phases: model loading (spinner), audio download (bar + ETA), and transcription (bar + ETA). Batch/multi-file runs show a second file-level bar alongside the per-file bar. All output is gated on TTY detection so piped output is unaffected.

## New module: `progress.py`

Three public functions only — no classes, no state objects.

### `spinner_start(msg: str) -> threading.Thread`

Starts a background daemon thread that overwrites the current line every 80ms, cycling through the braille spinner frames `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`. Output format: `\r⠸ {msg}`. Returns the thread handle for later stopping. No-ops (returns a do-nothing thread) if `not sys.stdout.isatty()`.

### `spinner_stop(thread: threading.Thread) -> None`

Sets a stop event on the thread and joins it, then writes `\r` + space-clear to erase the spinner line. Safe to call even when the thread was a no-op.

### `draw_bar(label: str, pct: float, eta_str: str = "") -> None`

Writes a single `\r`-overwritten line to stdout:

```
{label}  [████████████░░░░░░░░]  73%  0:12 remaining
```

- `label` is left-padded to a fixed width (12 chars) so the two bars in batch mode stay vertically aligned.
- Bar width is 20 cells; fill character `█`, empty character `░`.
- `pct` is clamped to 0–100.
- `eta_str` is appended if non-empty; omitted otherwise.
- No-ops if `not sys.stdout.isatty()`.

ETA string format is `M:SS remaining` (e.g., `0:12 remaining`, `1:04 remaining`). Callers compute the value; `draw_bar` just renders it.

## Integration points

### Model loading — `transcribe.py` (all three `run_*_workflow` functions)

Wrap the `WhisperModel(...)` constructor call:

```python
thread = spinner_start(f"Loading model {args.model}…")
model = WhisperModel(args.model, device=device, compute_type=compute_type)
spinner_stop(thread)
```

The spinner runs on a background daemon thread while the blocking constructor loads weights. After `spinner_stop` the existing `logger.info` lines resume normally.

### Audio download — `downloader.py`

Pass a `progress_hook` to yt-dlp that calls `draw_bar`. The hook receives a dict with `status`, `downloaded_bytes`, and `total_bytes` (or `total_bytes_estimate`). When `status == "downloading"` and total is known, compute `pct = downloaded / total * 100` and an ETA from elapsed time. When `status == "finished"` print a newline to terminate the overwritten line.

```python
def _make_progress_hook():
    start = time.time()
    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = d["downloaded_bytes"] / total * 100
                elapsed = time.time() - start
                remaining = (elapsed / max(pct, 0.1)) * (100 - pct)
                draw_bar("Downloading", pct, _fmt_eta(remaining))
        elif d["status"] == "finished":
            print()
    return hook
```

### Transcription progress — `transcribe_file` in `transcribe.py`

Replace the existing `\rTranscribing {pct}%` block with `draw_bar`. Add an optional `file_progress: tuple[int, int] | None = None` parameter (1-indexed `(current, total)`) for batch callers to pass file-level position.

ETA calculation: track `start_time` before the segment loop (already exists). On each segment: `elapsed = time.time() - start_time`, `remaining = elapsed / max(pct/100, 0.001) - elapsed`, then `_fmt_eta(remaining)`.

When `file_progress` is provided, the two bars require a separate function — `draw_two_bars(pct, eta_str, file_index, file_total)` — which manages cursor movement internally:

- **First render:** print line 1 + `\n` + line 2 (cursor is now on line 2).
- **Subsequent renders:** emit `\033[2A` (cursor up 2) then reprint both lines.
- A module-level `_two_bar_initialized: bool` flag (reset to `False` by `reset_two_bars()`) distinguishes the first render from subsequent ones.
- `reset_two_bars()` also emits a final `\n` to leave the cursor below the last bar.

```
Transcribing  [████████████░░░░░░░░]  73%  0:12 remaining
Files         [████░░░░░░░░░░░░░░░░]   1/3  file 1 of 3
```

Batch workflows call `reset_two_bars()` at the start of each new file to clear the flag. Single-file mode never calls `draw_two_bars` — it uses `draw_bar` directly.

### Batch callers — `run_project_workflow` and `run_multi_file_workflow`

Pass `file_progress=(i, len(urls_or_paths))` when calling `transcribe_file`. Single-file workflow passes `None`.

## Helper: `_fmt_eta(seconds: float) -> str`

Private to `progress.py`. Converts a float number of seconds into `M:SS remaining`. Returns empty string if seconds is negative or unreasonably large (> 1 hour, as a sanity guard).

## Degradation

- All three functions check both `sys.stdout.isatty()` **and** `logging.getLogger("transcribe").isEnabledFor(logging.INFO)` at call time. The `isatty()` check handles piped output and CI; the log-level check ensures `-q` (quiet mode) suppresses progress output. Both conditions must be true for any output to appear.
- The existing `_progress_enabled()` helper in `transcribe.py` is removed; its logic moves into a module-private `_should_render()` in `progress.py` and is called by `draw_bar` and `spinner_start`.

## Files changed

| File | Change |
|---|---|
| `src/transcribe/progress.py` | **New.** `spinner_start`, `spinner_stop`, `draw_bar`, `_fmt_eta` |
| `src/transcribe/transcribe.py` | Replace progress block; add `file_progress` param; import from `progress` |
| `src/transcribe/downloader.py` | Add `_make_progress_hook`; pass to yt-dlp |
| `src/transcribe/cli.py` | No change needed (workflow functions live in `transcribe.py`) |
| `.gitignore` | `.superpowers/` already added |

## Out of scope

- Color theming or configurable bar style
- Progress persistence across resumed runs
- Windows / non-ANSI terminal support (WSL2 with a modern terminal is the target environment)
