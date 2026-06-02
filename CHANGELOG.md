# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-02

### Added
- `--version` flag (reports the package version).
- `--list-models` flag (prints the curated list of Whisper models and exits).
- `--json` flag: writes a parallel `.json` transcript alongside `.txt`/`.srt` containing language, duration, and segment list (with optional `words` when `--word-timestamps` is used). Coexists with other output formats.
- `word_timestamps` support: new config key (`TRANSCRIBE_WORD_TIMESTAMPS`), `--word-timestamps` flag, and passthrough to `faster-whisper`. Follows the exact 4-step process documented in CLAUDE.md. Richer segment data is available for `--json` (and future uses).
- Optional `[cuda]` extra for `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` wheels. CPU-only is now the default (lighter installs via `uv tool install .`); GPU users opt in with `.[cuda]` or `transcribe[cuda]`. `resolve_cuda_libs` / `cuda_available` / `default_device` remain fully graceful.
- `ffmpeg_available()` helper in utils; early actionable checks + error hints for missing `ffmpeg` (critical for URL/project downloads via yt-dlp).
- Improved error messages/hints for CUDA/GPU failures (suggest `--device cpu` etc.).
- GitHub Actions CI workflow (`.github/workflows/ci.yml`): Linux matrix for Python 3.11 + 3.12, runs ruff + format check + pytest (CPU path only).
- `CHANGELOG.md` (this file).
- New unit tests for parser flags, `write_json`, config key (env/merge/template), and the progress renderer class.

### Changed
- Progress bar internals: introduced private `_TwoBarRenderer` class (plus `_two_bar_state` dict singleton) to encapsulate the previous module-global `_two_bar_initialized` flag and cursor-up logic. Public API (`draw_two_bars`, `reset_two_bars`, etc.) and behavior (TTY gating, ETA, spinners) are unchanged. This eliminates `PLW0603` lint and improves testability/maintainability (addresses findings from 2026 review static analysis).
- nvidia CUDA wheels moved from hard `[project]dependencies` to optional `[project.optional-dependencies] cuda`.
- Version bumped to 0.2.0 (new features + packaging + internal refactor since 0.1.0). `__version__` and pyproject now in sync; `--version` wired.
- Docs refreshed (README install matrix + examples for all new flags, troubleshooting for ffmpeg/GPU; CLAUDE.md module map + new-key process example + review notes).
- Some long lines and minor ruff findings cleaned as part of the work (default `ruff check` now passes cleanly).
- Resume logic in batch/project workflows now also considers the corresponding `.json` when `--json` is used.

### Fixed
- The three pre-existing default-ruff issues (E501 help/comment, F841 unused var in test) from the initial review.
- Global statement lint for two-bar progress state.

## [0.1.0] - Initial

- Initial release (see git history and README for original feature set: faster-whisper wrapper, layered config, project mode, cookies, CUDA preload hack, progress bars+spinners, etc.).
