<div align="center">

![transcribe](assets/transcribe.png)

# transcribe

**GPU-aware audio/video transcription powered by `faster-whisper`**

![Python](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-22863a?logo=opensourceinitiative&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-lightgrey?logo=linux&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-auto--detect-76b900?logo=nvidia&logoColor=white)

</div>

---

<table>
<tr>
<td>🎙️ <b>Single files, batches, or URLs</b><br>One command handles them all</td>
<td>⚡ <b>GPU-aware out of the box</b><br>Auto-loads CUDA libs — no <code>LD_LIBRARY_PATH</code> gymnastics</td>
</tr>
<tr>
<td>📁 <b>Project mode</b><br>Download + transcribe video URLs into organized project folders</td>
<td>🔁 <b>Resume-safe batches</b><br>Skip completed transcripts; re-run the same command safely</td>
</tr>
<tr>
<td>🔧 <b>Layered config</b><br>CLI flags > env vars > TOML file > built-in defaults</td>
<td>📝 <b>TXT and SRT output</b><br>Plain transcript or subtitle-ready SRT, your choice</td>
</tr>
<tr>
<td>🍪 <b>Cookie authentication</b><br>Bypass YouTube bot detection via browser cookies or a cookies.txt file</td>
<td></td>
</tr>
</table>

---

## Contents

- [Install](#install)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Project mode — download and transcribe URLs](#project-mode--download-and-transcribe-urls)
  - [Batch transcribe local files](#batch-transcribe-local-files)
  - [Common examples](#common-examples)
- [Models](#models)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Install

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). For GPU mode you also need an
NVIDIA driver and `ffmpeg` (`sudo apt install ffmpeg`).

```bash
git clone <this-repo> ~/code/transcribe
cd ~/code/transcribe
uv tool install .
```

This puts a `transcribe` command on your PATH. For development use an editable install instead:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

GPU is auto-detected: when CUDA is present, `transcribe` locates `libcublas` and `libcudnn`
inside the installed `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels and loads them in-process.
No `LD_LIBRARY_PATH` setup needed.

---

## Configuration

<details>
<summary>Config file, env vars, and precedence</summary>

A TOML config is auto-created on first run at `$XDG_CONFIG_HOME/transcribe/config.toml`
(typically `~/.config/transcribe/config.toml`). All keys are commented out — uncomment any
line to override the built-in default for that key.

**Precedence** (lowest → highest): builtin default → config file → `TRANSCRIBE_*` env var → CLI flag.

Environment overrides use the prefix `TRANSCRIBE_` and match the config keys:

```bash
TRANSCRIBE_MODEL=small
TRANSCRIBE_DEVICE=cpu
TRANSCRIBE_PROJECT_ROOT=/data/transcripts
TRANSCRIBE_COOKIES_FROM_BROWSER=chrome
TRANSCRIBE_COOKIES_FILE=/path/to/cookies.txt
```

</details>

---

## Usage

Interactive mode (prompts for mode, model, and file/URLs):

```bash
transcribe
```

Single file:

```bash
transcribe ~/Downloads/example.m4a
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

Project folders are created under `./projects/` (relative to your working directory) by default.
Override with `--project-root /some/path`, the `TRANSCRIBE_PROJECT_ROOT` env var, or `project_root`
in the config file. The example above produces:

```
projects/my-research/my-research1.wav   ← extracted audio
projects/my-research/my-research1.txt   ← transcript
projects/my-research/my-research2.wav
projects/my-research/my-research2.txt
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

### Common examples

```bash
# Different model
transcribe --model large-v3

# Write both .txt and .srt
transcribe --srt

# Force GPU / CPU
transcribe --device cuda
transcribe --device cpu

# Use GPU 1 only
CUDA_VISIBLE_DEVICES=1 transcribe

# Delete audio files after transcription
transcribe --cleanup

# Custom output directory
transcribe ~/Downloads/example.m4a --output-dir ~/Downloads/transcripts

# Verbose / quiet logging
transcribe -v clip.wav   # DEBUG: timestamped, module-named log lines
transcribe -q clip.wav   # WARNING+ only; suppresses the progress line

# Bypass YouTube bot detection using browser cookies
transcribe --cookies-from-browser chrome https://youtu.be/abc123
transcribe --cookies-from-browser firefox https://youtu.be/abc123

# Use an exported cookies.txt file instead
transcribe --cookies ~/cookies.txt https://youtu.be/abc123
```

---

## Models

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `turbo` | ⚡⚡⚡ | ★★★★ | Best general speed/quality tradeoff |
| `small` | ⚡⚡⚡⚡ | ★★★ | Lighter and faster, lower accuracy |
| `medium` | ⚡⚡ | ★★★★ | Middle ground |
| `large-v3` | ⚡ | ★★★★★ | Best accuracy, slower and heavier |

---

## Troubleshooting

### YouTube: "Sign in to confirm you're not a bot"

YouTube increasingly blocks unauthenticated yt-dlp downloads. Pass your browser's cookie
session to authenticate the download:

```bash
# Read cookies from an installed browser (chrome, firefox, edge, chromium, safari)
transcribe --cookies-from-browser chrome https://youtu.be/abc123

# Or set it once in ~/.config/transcribe/config.toml so you never need the flag:
# cookies_from_browser = "chrome"
```

If browser cookie extraction fails (e.g. in headless environments), export a `cookies.txt`
from your browser using an extension like [Get cookies.txt LOCALLY](https://github.com/kairi003/Get-cookies.txt-LOCALLY)
and pass it with `--cookies ~/cookies.txt`.

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

---

## Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
ruff format src/ tests/
```

---

<div align="center">

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [uv](https://docs.astral.sh/uv/) &nbsp;|&nbsp; MIT License

</div>
