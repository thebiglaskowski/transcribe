# faster-whisper transcription in WSL2 with uv

This setup uses `faster-whisper` to transcribe an audio file and save a `.txt` transcript.

## What this does

- prompts you for a file path if you do not pass one on the command line
- writes a `.txt` transcript next to the source file
- can also write an `.srt` subtitle file if you use `--srt`
- uses GPU automatically if CUDA is available in WSL

## 1) Create and activate your venv

```bash
mkdir -p ~/.transcribe
cd ~/.transcribe
uv venv --python 3.11
source .venv/bin/activate
```

## 2) Install the packages

```bash
uv pip install -U faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

## 3) Export the CUDA library path

Run this in your shell before using the script:

```bash
export TRANSCRIBE_SITEPKG="/home/joe/.transcribe/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$TRANSCRIBE_SITEPKG/nvidia/cublas/lib:$TRANSCRIBE_SITEPKG/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

To make it permanent, add those two lines to `~/.bashrc`, then run:

```bash
source ~/.bashrc
```

## 4) Save the script

Save `transcribe_audio.py` in `~/.transcribe`.

## 5) Run it

Interactive mode:

```bash
python transcribe_audio.py
```

It will prompt you for the file path, for example:

```text
/mnt/c/Users/joela/Downloads/example.m4a
```

Direct file path mode:

```bash
python transcribe_audio.py /mnt/c/Users/joela/Downloads/example.m4a
```

## Project mode — download and transcribe URLs

Pass one or more video URLs and the script enters project mode: it prompts you for a project name, creates a folder for it, downloads and extracts audio via yt-dlp, then transcribes each video in sequence.

```bash
python transcribe_audio.py https://youtu.be/abc123 https://youtu.be/def456
```

You will be prompted:

```
Project name (a folder will be created): my-research
```

This creates `~/.transcribe/my-research/` and produces:

```
my-research/my-research1.wav   ← extracted audio (kept for re-transcription)
my-research/my-research1.txt   ← transcript
my-research/my-research2.wav
my-research/my-research2.txt
...
```

Works with YouTube, TikTok, Instagram, and any site yt-dlp supports. Requires ffmpeg to be installed (`sudo apt install ffmpeg`).

## Batch transcribe local files

Pass multiple local file paths to transcribe them all in one run without downloading anything:

```bash
python transcribe_audio.py clip1.wav clip2.m4a clip3.mp4
```

Transcripts are saved next to each source file. Use `--output-dir` to redirect all output to a single folder:

```bash
python transcribe_audio.py clip1.wav clip2.m4a --output-dir ./transcripts
```

## Common examples

Default run:

```bash
python transcribe_audio.py
```

Use a different model:

```bash
python transcribe_audio.py --model large-v3
```

Write both `.txt` and `.srt`:

```bash
python transcribe_audio.py --srt
```

Force GPU:

```bash
python transcribe_audio.py --device cuda
```

Force CPU:

```bash
python transcribe_audio.py --device cpu
```

Use GPU 1 only:

```bash
CUDA_VISIBLE_DEVICES=1 python transcribe_audio.py
```

Save output somewhere else:

```bash
python transcribe_audio.py /mnt/c/Users/joela/Downloads/example.m4a --output-dir /mnt/c/Users/joela/Downloads/transcripts
```

## Model suggestions

- `turbo`: best general speed/quality tradeoff
- `small`: lighter and faster, lower accuracy
- `medium`: middle ground
- `large-v3`: best accuracy, slower and heavier

## Troubleshooting

### `libcublas.so.12 is not found`

Your CUDA library path is not set in the shell that launched Python. Re-run:

```bash
export TRANSCRIBE_SITEPKG="/home/joe/.transcribe/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$TRANSCRIBE_SITEPKG/nvidia/cublas/lib:$TRANSCRIBE_SITEPKG/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

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

## Notes

- Do not use `uv run` with inline script metadata for this particular setup.
- Run the script from your activated venv with `python transcribe_audio.py`.
- `faster-whisper` will download the model on first use.
