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

This puts a `transcribe` command on your PATH (CPU-only by default; see below for GPU).

For development use an editable install instead:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**GPU support (optional):** The heavy NVIDIA CUDA wheels are not installed by default (keeps CPU-only installs small). To enable GPU:

```bash
uv tool install '.[cuda]'
# or for development:
uv pip install -e ".[dev,cuda]"
```

GPU is auto-detected: when the `[cuda]` extra is installed *and* CUDA is present, `transcribe` locates `libcublas` and `libcudnn`
inside the installed `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels and loads them in-process.
No `LD_LIBRARY_PATH` setup needed. If you run on CPU only, just omit the `[cuda]` extra.

**Speaker labels (optional):** `--diarize` (or answering *y* to "Label speakers?") tags each
paragraph `Speaker 1:`, `Speaker 2:`… using [pyannote](https://hf.co/pyannote/speaker-diarization-community-1).
It pulls in torch (several GB), so it's a separate extra, and the model is gated on Hugging Face:

```bash
uv tool install --force --reinstall '.[cuda,diarize]'
# once: accept the terms at https://hf.co/pyannote/speaker-diarization-community-1,
# create a Read token at https://hf.co/settings/tokens, then:
uvx --from huggingface_hub hf auth login
```

torch is pinned to its CUDA 12 build (see `[tool.uv.sources]`) so it shares faster-whisper's cuDNN —
the default CUDA 13 torch would corrupt it. Speakers are numbered in order of first appearance — find/replace
`Speaker 1:` with a real name afterwards. Set `diarize = true` in the config to default the prompt to yes.

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

**Whole channels work too.** Paste a YouTube channel (`youtube.com/@name`), a playlist, or a
TikTok profile (`tiktok.com/@name`). For a YouTube channel you pick which tabs you want —
Videos, Live, Shorts. Every source (each tab, playlist or profile) gets its own project folder
and its own "how many, newest first?" question. All questions come first, then the run goes
unattended. Re-running the same channel later only transcribes new uploads. Plain video URLs
pasted alongside go into one shared folder, as before.

```bash
transcribe https://youtu.be/abc123 https://youtu.be/def456
```

You will be prompted:

```
https://www.youtube.com/@gg33academy has:
  [1] GG33 - Videos (288 videos)
  [2] GG33 - Live (212 videos)
  [3] GG33 - Shorts (2410 videos)
Which? e.g. 1 or 1,2 (Enter for all): 1,2

GG33 - Videos: how many, newest first? [all 288, 0 skips]
GG33 - Live: how many, newest first? [all 212, 0 skips] 20
gg33academy: how many, newest first? [all 1893, 0 skips] 0
Project name [GG33-Videos] (a folder will be created):
Project name [GG33-Live] (a folder will be created):
```

Project folders are created under `./projects/` (relative to your working directory) by default.
Override with `--project-root /some/path`, the `TRANSCRIBE_PROJECT_ROOT` env var, or `project_root`
in the config file. Files are named after the video's title plus its id:

```
projects/Veritasium/Is spider web really stronger than steel [wt4p2oalmRY].txt
projects/Veritasium/Why does every mammal get 1 billion heartbeats in their life [tL9Lw250spc].txt
...
```

Each transcript opens with the video's title, channel, upload date and URL above a `---` line.
The id keeps names unique and makes resume reliable: a video whose transcript already exists is
skipped, however the channel's order has shifted. Videos with no speech get a
`[no speech detected]` placeholder so re-runs don't retry them. (Folders from before this
naming scheme used `name1.txt`, `name2.txt`… and won't be recognised as done.)

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

# Write structured JSON (coexists with txt/srt; includes words if --word-timestamps)
transcribe --json clip.mp3
transcribe --json --word-timestamps clip.mp3

# Enable word-level timestamps (richer data for --json or future use)
transcribe --word-timestamps clip.mp3

# Label speakers (needs the [diarize] extra — see Install)
transcribe --diarize --srt interview.mp3

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

# Version and model list
transcribe --version
transcribe --list-models

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

The `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels are missing from your install (they live in the optional `[cuda]` extra). Reinstall with the GPU extra:

```bash
uv tool install --reinstall '.[cuda]'
# or for an editable dev install:
uv pip install -e --reinstall ".[dev,cuda]"
```

If you intend to run on CPU only, simply omit the `[cuda]` extra (the default `uv tool install .` is now CPU-only and much lighter). You can still force CPU with `--device cpu` or `device = "cpu"` in `~/.config/transcribe/config.toml`.

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

### Missing ffmpeg (for URL / project downloads)

Project mode (YouTube etc.) uses yt-dlp which requires `ffmpeg` for audio extraction.

Error will now suggest: `sudo apt install ffmpeg`

The `transcribe` CLI also checks early when URLs are passed.

### GPU / CUDA failures

If model load fails with CUDA errors, the error message now suggests `--device cpu` (or ensure the `[cuda]` extra + NVIDIA driver + `nvidia-smi` works).

You can always force CPU: `transcribe --device cpu ...` (no `[cuda]` extra needed).

---

## Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                       # ~49 pure unit tests (no audio fixtures)
ruff check src/ tests/
ruff format src/ tests/

CI (GitHub Actions) runs the equivalent on ubuntu for py 3.11 + 3.12 (CPU path).
```

---

<div align="center">

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [uv](https://docs.astral.sh/uv/) &nbsp;|&nbsp; MIT License

</div>
