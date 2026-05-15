# transcribe

GPU-aware audio/video transcription with `faster-whisper`. Wraps the Whisper model with a
CLI that handles single files, batches of local files, and project folders built from
video URLs (YouTube, TikTok, etc. via `yt-dlp`).

## Install

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). For GPU mode you also need an
NVIDIA driver and `ffmpeg` (`sudo apt install ffmpeg`).

```bash
git clone <this-repo> ~/code/transcribe
cd ~/code/transcribe
uv tool install .
```

This puts a `transcribe` command on your PATH. For development use an editable install
instead:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

GPU is auto-detected: when CUDA is present, `transcribe` locates `libcublas` and `libcudnn`
inside the installed `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels and loads them in-process.
No `LD_LIBRARY_PATH` setup needed.

## Configuration

A TOML config is auto-created on first run at `$XDG_CONFIG_HOME/transcribe/config.toml`
(typically `~/.config/transcribe/config.toml`). All keys are commented out — uncomment any
line to override the built-in default for that key.

Precedence (lowest to highest): builtin default → config file → `TRANSCRIBE_*` env var → CLI flag.

Environment overrides use the prefix `TRANSCRIBE_` and match the config keys, e.g.
`TRANSCRIBE_MODEL=small`, `TRANSCRIBE_DEVICE=cpu`, `TRANSCRIBE_PROJECT_ROOT=/data/transcripts`.

## Usage

Interactive mode (prompts for mode, model, and file/URLs):

```bash
transcribe
```

Single file:

```bash
transcribe /mnt/c/Users/joela/Downloads/example.m4a
```

### Project mode — download and transcribe URLs

Pass one or more video URLs and the script enters project mode: it prompts you for a
project name, creates a folder for it, downloads and extracts audio via `yt-dlp`, then
transcribes each video in sequence.

```bash
transcribe https://youtu.be/abc123 https://youtu.be/def456
```

You will be prompted:

```
Project name (a folder will be created): my-research
```

Project folders are created under `~/.transcribe/projects/` by default. Override with
`--project-root /some/path`, the `TRANSCRIBE_PROJECT_ROOT` env var, or `project_root` in
the config file. The example above produces:

```
~/.transcribe/projects/my-research/my-research1.wav   ← extracted audio
~/.transcribe/projects/my-research/my-research1.txt   ← transcript
~/.transcribe/projects/my-research/my-research2.wav
~/.transcribe/projects/my-research/my-research2.txt
...
```

Works with any site `yt-dlp` supports.

### Batch transcribe local files

```bash
transcribe clip1.wav clip2.m4a clip3.mp4
```

Transcripts are saved next to each source file. If a transcript already exists for a
file it will be skipped, so interrupted batch runs can be safely resumed by re-running
the same command.

Use `--output-dir` to redirect all output to a single folder:

```bash
transcribe clip1.wav clip2.m4a --output-dir ./transcripts
```

## Common examples

Different model:

```bash
transcribe --model large-v3
```

Write both `.txt` and `.srt`:

```bash
transcribe --srt
```

Force GPU / CPU:

```bash
transcribe --device cuda
transcribe --device cpu
```

Use GPU 1 only:

```bash
CUDA_VISIBLE_DEVICES=1 transcribe
```

Delete audio files after transcription:

```bash
transcribe --cleanup
```

Custom output directory:

```bash
transcribe ~/Downloads/example.m4a --output-dir ~/Downloads/transcripts
```

Verbose / quiet logging:

```bash
transcribe -v clip.wav   # DEBUG: timestamped, module-named log lines
transcribe -q clip.wav   # WARNING+ only; suppresses the progress line
```

## Model suggestions

- `turbo` — best general speed/quality tradeoff
- `small` — lighter and faster, lower accuracy
- `medium` — middle ground
- `large-v3` — best accuracy, slower and heavier

## Troubleshooting

### `libcublas.so.12 is not found`

The `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels are missing from your install. Reinstall
with `uv tool install --reinstall .`. If you intend to run on CPU only, pass `--device cpu` or
set `device = "cpu"` in `~/.config/transcribe/config.toml`.

### Check that WSL sees your GPU

```bash
nvidia-smi
```

### Confirm the CUDA libraries are loadable

This runs the same auto-resolution `transcribe` does on startup and prints whether GPU is
available:

```bash
python - <<'PY'
from transcribe.utils import cuda_available, resolve_cuda_libs
resolve_cuda_libs()
print("GPU available:", cuda_available())
PY
```

## Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
ruff format src/ tests/
```
