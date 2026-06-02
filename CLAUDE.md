# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small installable Python package that wraps `faster-whisper` with a CLI. Source lives under `src/transcribe/`, exposed as the `transcribe` console script. The user installs with `uv tool install .` (or `uv pip install -e ".[dev]"` for development).

## Install / dev loop

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
# For GPU: uv pip install -e ".[dev,cuda]"
pytest                       # ~49 unit tests (pure functions only; grew with progress bars), no audio fixtures required
ruff check src/ tests/
ruff format src/ tests/
```

There is no end-to-end test fixture in-repo — verification with real audio is manual.

## Architecture (the parts that need orientation)

The `cli.main` flow is: load config → parse args → configure logging → resolve CUDA libs → dispatch.

- **Layered config** (`config.py`). Precedence is builtin defaults < `~/.config/transcribe/config.toml` < `TRANSCRIBE_*` env vars < CLI flags. The first three layers are merged into a `dict` and fed into `build_parser` so argparse uses them as defaults — that's why the CLI flag always wins. The TOML file is auto-created (commented template) on first run.
- **`--model` is a deliberate `None` sentinel.** `build_parser` does NOT seed `--model` from the merged config, because `cli.main` needs to distinguish "user passed --model" from "default applied" to decide whether the interactive model menu should fire. After parsing, `cli.main` resolves the final value with explicit precedence checks (see the comment near `cli_supplied_model`).
- **Dispatch by input shape** (`cli._classify_inputs`): URLs only → `run_project_workflow` (downloads via `yt-dlp`, creates a project subfolder under `--project-root`). All paths → `run_single_file_workflow` or `run_multi_file_workflow`. Mixed inputs are rejected.
- **CUDA without `LD_LIBRARY_PATH`** (`utils.resolve_cuda_libs`). Called from `cli.main` before model load. Imports `nvidia.cublas.lib` and `nvidia.cudnn.lib` to find the wheel directories, prepends them to `LD_LIBRARY_PATH`, then `ctypes.CDLL`-preloads `libcublas.so.12` and `libcudnn.so.9` with `RTLD_GLOBAL` so faster-whisper's later dlopen resolves against them. No-op if the wheels are missing (the wheels now live in the optional `[cuda]` extra — see Install / dev loop). Graceful for CPU-only installs (`uv tool install .` or `uv pip install -e ".[dev]"`); GPU users add `[cuda]`. The old "hard dependency" comment is outdated post-0.2.0.
- **Resume semantics** are still "transcript file exists and is non-empty → skip." Preserved verbatim in both batch workflows. Re-running the same command is the supported resume mechanism.

## Module map

- `cli.py` — argparse, logging setup, dispatch. No business logic.
- `config.py` — `Settings` dataclass, TOML loader, env parser, `merge`.
- `transcribe.py` — `transcribe_file` + the three `run_*_workflow` functions. Workflows currently take `argparse.Namespace`; this is a known wart (would prefer a typed config object passed through).
- `prompts.py` — every interactive `input()` lives here. These intentionally use `print`, not `logging`, because they're conversational I/O.
- `output.py` — `write_txt`, `write_srt`.
- `downloader.py` — yt-dlp wrapper, with `import yt_dlp` lazy inside the function (so non-URL invocations don't pay its import cost).
- `utils.py` — `is_url`, `srt_timestamp`, `sanitize_project_name`, `SUPPORTED_EXTENSIONS`, CUDA helpers.
- `progress.py` — TTY-gated ANSI bars + ETA + spinners + two-bar batch mode (originally no-classes design; 2026 review refactored to small TwoBarRenderer class to fix globals/PLW0603/fragility). Gated on `sys.stdout.isatty() and logger.isEnabledFor(INFO)`.

## Logging

Every non-prompt module owns a module-named logger (`logging.getLogger(__name__)`). `cli._configure_logging` configures the root logger once based on `-v` / `-q`. Default INFO format is bare `%(message)s`; DEBUG includes timestamp, level, and module. The carriage-return progress line in `transcribe_file` stays as direct `print()` but is gated on `logger.isEnabledFor(logging.INFO)` so `-q` silences it.

## Things to know before changing things

- The `.gitignore` is a conventional Python denylist now (it used to be allow-list — pre-refactor history will look weird if you bisect).
- Tests deliberately do not mock `WhisperModel`; they cover pure functions only.
- If you add a new config key, you need to: (1) add it to `Settings`, (2) add a `TRANSCRIBE_*` entry to `_ENV_MAP`, (3) add a `parser.add_argument` line with `default=defaults["new_key"]`, (4) optionally extend the commented template in `DEFAULT_CONFIG_TEMPLATE`.
  Example (word_timestamps added in 0.2.0): see Settings, _ENV_MAP (TRANSCRIBE_WORD_TIMESTAMPS), cli.py add_argument, and DEFAULT_CONFIG_TEMPLATE.
- yt-dlp's import is lazy by design — don't move it to the top of `downloader.py`.
- 2026 project review (static analysis via radon/ruff/coverage/mypy/vulture + MCP CodeGraphContext indexing + explore subagent) identified the exact items in this plan (globals in progress, hard nvidia deps, no --version/CI, doc drift, etc.). See session plan.md for full findings + job IDs/timeouts.
