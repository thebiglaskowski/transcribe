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

### GPU library path (temporary)

Until the next release auto-resolves it, GPU users must still export the CUDA library
path to the venv where `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` were installed. With
`uv tool install`, that location is under `~/.local/share/uv/tools/transcribe/`:

```bash
export TRANSCRIBE_SITEPKG="$(uv tool dir)/transcribe/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$TRANSCRIBE_SITEPKG/nvidia/cublas/lib:$TRANSCRIBE_SITEPKG/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

Add to `~/.bashrc` to make permanent.

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

This creates a `my-research/` folder and produces:

```
my-research/my-research1.wav   ← extracted audio
my-research/my-research1.txt   ← transcript
my-research/my-research2.wav
my-research/my-research2.txt
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

## Model suggestions

- `turbo` — best general speed/quality tradeoff
- `small` — lighter and faster, lower accuracy
- `medium` — middle ground
- `large-v3` — best accuracy, slower and heavier

## Troubleshooting

### `libcublas.so.12 is not found`

Your CUDA library path is not set. See the "GPU library path" section above.

### Check that WSL sees your GPU

```bash
nvidia-smi
```

### Confirm the CUDA libraries are loadable

```bash
python - <<'PY'
import ctypes
ctypes.CDLL("libcublas.so.12")
ctypes.CDLL("libcudnn.so.9")
print("CUDA libraries loaded OK")
PY
```
