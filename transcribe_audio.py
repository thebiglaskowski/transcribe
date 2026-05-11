from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel

SUPPORTED_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
}

DEFAULT_MODEL = "turbo"

MODELS = [
    ("turbo",    "Turbo     — fast, great accuracy (recommended)"),
    ("large-v3", "Large v3  — highest accuracy, slowest"),
    ("medium",   "Medium    — balanced speed / accuracy"),
    ("small",    "Small     — fast, lighter RAM"),
    ("tiny",     "Tiny      — fastest, lowest accuracy"),
]
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD = True
DEFAULT_SRT = False

SCRIPT_DIR = Path(__file__).parent


def cuda_available() -> bool:
    try:
        ctypes.CDLL("libcublas.so.12")
        ctypes.CDLL("libcudnn.so.9")
        return True
    except OSError:
        return False


def default_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def default_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "int8"


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def prompt_project_name() -> tuple[str, Path]:
    while True:
        raw = input("Project name (a folder will be created): ").strip()
        if not raw:
            print("Name cannot be empty.\n")
            continue
        safe = re.sub(r"[^\w\-]", "-", raw).strip("-")
        if not safe:
            print("Name produced no valid characters after sanitizing. Try again.\n")
            continue
        project_dir = SCRIPT_DIR / safe
        if project_dir.exists():
            confirm = input(f"Folder '{safe}' already exists. Use it? [y/N] ").strip().lower()
            if confirm != "y":
                continue
        project_dir.mkdir(parents=True, exist_ok=True)
        return safe, project_dir


def download_audio(url: str, project_dir: Path, stem: str) -> Path:
    """Download audio from a URL using yt-dlp. Returns the path to the extracted audio file."""
    import yt_dlp

    outtmpl = str(project_dir / f"{stem}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": False,
        "no_warnings": False,
    }

    print(f"\nDownloading: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = project_dir / f"{stem}.wav"
    if not audio_path.exists():
        # yt-dlp may keep original extension in some cases — look for any match
        candidates = list(project_dir.glob(f"{stem}.*"))
        audio_candidates = [p for p in candidates if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not audio_candidates:
            raise FileNotFoundError(f"Could not find downloaded audio for stem '{stem}' in {project_dir}")
        audio_path = audio_candidates[0]

    return audio_path


def prompt_model_menu() -> str:
    """Return the selected Whisper model name."""
    print("\nWhich model would you like to use?")
    for i, (_, label) in enumerate(MODELS, start=1):
        print(f"  [{i}] {label}")
    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            return MODELS[int(choice) - 1][0]
        print(f"Enter a number between 1 and {len(MODELS)}.\n")


def prompt_mode_menu() -> str:
    """Return 'single', 'urls', or 'batch'."""
    print("\nWhat would you like to do?")
    print("  [1] Transcribe a local audio/video file")
    print("  [2] Download and transcribe video URLs  (project mode)")
    print("  [3] Batch transcribe multiple local files")
    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice == "1":
            return "single"
        if choice == "2":
            return "urls"
        if choice == "3":
            return "batch"
        print("Enter 1, 2, or 3.\n")


def prompt_for_urls() -> list[str]:
    """Interactively collect one or more video URLs."""
    print("\nEnter video URLs, one per line. Press Enter on a blank line when done.")
    urls: list[str] = []
    while True:
        line = input(f"  URL {len(urls) + 1}: ").strip()
        if not line:
            if not urls:
                print("At least one URL is required.\n")
                continue
            break
        if not is_url(line):
            print(f"  Not a recognised URL (must start with http:// or https://). Try again.")
            continue
        urls.append(line)
    return urls


def prompt_for_batch_files() -> list[str]:
    """Interactively collect one or more local file paths."""
    print("\nEnter file paths, one per line. Press Enter on a blank line when done.")
    paths: list[str] = []
    while True:
        line = input(f"  File {len(paths) + 1}: ").strip().strip('"').strip("'")
        if not line:
            if not paths:
                print("At least one file is required.\n")
                continue
            break
        p = Path(line).expanduser()
        if not p.exists():
            print(f"  File not found: {p}")
            continue
        if not p.is_file():
            print(f"  Not a file: {p}")
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"  Unsupported extension '{p.suffix}'. Allowed: {allowed}")
            continue
        paths.append(str(p))
    return paths


def prompt_for_audio_file() -> Path:
    while True:
        raw = input("Enter full path to the audio file: ").strip().strip('"').strip("'")
        if not raw:
            print("No path entered. Try again.\n")
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"File not found: {path}\n")
            continue
        if not path.is_file():
            print(f"Not a file: {path}\n")
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"Unsupported extension: {path.suffix}\nAllowed: {allowed}\n")
            continue
        return path


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_txt(segments: Iterable, output_path: Path) -> None:
    parts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            parts.append(text)
    transcript = "\n\n".join(parts).strip()
    output_path.write_text((transcript + "\n") if transcript else "", encoding="utf-8")


def write_srt(segments: list, output_path: Path) -> None:
    lines: list[str] = []
    index = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        index += 1
        start = srt_timestamp(segment.start)
        end = srt_timestamp(segment.end)
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def transcribe_file(
    audio_path: Path,
    txt_path: Path,
    srt_path: Path | None,
    model: WhisperModel,
    args: argparse.Namespace,
    info_prefix: str = "",
) -> bool:
    """Transcribe a single audio file. Returns True on success."""
    print(f"\n{info_prefix}Input:  {audio_path}")
    print(f"{info_prefix}Output: {txt_path}")
    if srt_path:
        print(f"{info_prefix}SRT:    {srt_path}")

    start_time = time.time()
    use_vad = not args.no_vad

    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=args.beam_size,
        vad_filter=use_vad,
        language=args.language,
    )

    segments: list = []
    print("Transcribing ", end="", flush=True)
    for segment in segments_iter:
        segments.append(segment)
        if info.duration:
            pct = min(100, int(segment.end / info.duration * 100))
            print(f"\rTranscribing {pct:3d}%", end="", flush=True)
        else:
            print(".", end="", flush=True)
    print()

    if not segments:
        print("No transcript text was produced.")
        return False

    write_txt(segments, txt_path)
    if srt_path:
        write_srt(segments, srt_path)

    elapsed = time.time() - start_time
    print(f"Language: {info.language} ({info.language_probability:.2f}) | {elapsed:.1f}s")
    print(f"Saved: {txt_path}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio/video with faster-whisper.\n"
            "Pass one or more video URLs to enter project mode (downloads + transcribes each).\n"
            "Pass a local file path (or nothing) for single-file mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Video URL(s) or a local audio/video file path. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model. Examples: turbo, small, medium, large-v3. Default: turbo",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use. Default: auto",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Override compute type. Examples: float16, int8_float16, int8.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size. Default: {DEFAULT_BEAM_SIZE}",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable silence filtering (VAD).",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        default=DEFAULT_SRT,
        help="Also write an .srt subtitle file.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Source language code (e.g. en, es, fr). Default: auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for single-file mode. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Delete audio files after successful transcription.",
    )
    return parser


def run_project_workflow(urls: list[str], args: argparse.Namespace) -> int:
    project_name, project_dir = prompt_project_name()

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)

    print(f"\nLoading model: {args.model}")
    print(f"Device: {device} | Compute type: {compute_type}")
    print(f"Project folder: {project_dir}")
    print(f"Videos to process: {len(urls)}")

    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    successes = 0
    failures = 0

    for i, url in enumerate(urls, start=1):
        stem = f"{project_name}{i}"
        print(f"\n{'='*60}")
        print(f"[{i}/{len(urls)}] {url}")
        print(f"{'='*60}")

        txt_path = project_dir / f"{stem}.txt"
        srt_path = (project_dir / f"{stem}.srt") if args.srt else None

        if txt_path.exists() and txt_path.stat().st_size > 0:
            print(f"  Skipping — transcript already exists: {txt_path.name}")
            successes += 1
            continue

        try:
            audio_path = download_audio(url, project_dir, stem)
        except Exception as exc:
            print(f"Download failed: {exc}")
            failures += 1
            continue

        try:
            ok = transcribe_file(audio_path, txt_path, srt_path, model, args, info_prefix="  ")
            if ok:
                successes += 1
                if args.cleanup:
                    audio_path.unlink()
                    print(f"Deleted: {audio_path}")
            else:
                failures += 1
        except Exception as exc:
            print(f"Transcription failed: {exc}")
            failures += 1

    print(f"\n{'='*60}")
    print(f"Done. {successes} succeeded, {failures} failed.")
    print(f"Transcripts saved in: {project_dir}")
    return 0 if failures == 0 else 1


def run_multi_file_workflow(raw_paths: list[str], args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for raw in raw_paths:
        p = Path(raw.strip().strip('"').strip("'")).expanduser()
        if not p.exists():
            print(f"File not found: {p}")
            return 1
        if not p.is_file():
            print(f"Not a file: {p}")
            return 1
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"Unsupported extension: {p.suffix}. Allowed: {allowed}")
            return 1
        paths.append(p)

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)

    print(f"\nLoading model: {args.model}")
    print(f"Device: {device} | Compute type: {compute_type}")
    print(f"Files to process: {len(paths)}")

    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    output_dir_override = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir_override:
        output_dir_override.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0

    for i, audio_path in enumerate(paths, start=1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(paths)}] {audio_path.name}")
        print(f"{'='*60}")
        output_dir = output_dir_override or audio_path.parent
        txt_path = output_dir / f"{audio_path.stem}.txt"
        srt_path = (output_dir / f"{audio_path.stem}.srt") if args.srt else None

        if txt_path.exists() and txt_path.stat().st_size > 0:
            print(f"  Skipping — transcript already exists: {txt_path.name}")
            successes += 1
            continue

        try:
            ok = transcribe_file(audio_path, txt_path, srt_path, model, args, info_prefix="  ")
            if ok:
                successes += 1
                if args.cleanup:
                    audio_path.unlink()
                    print(f"Deleted: {audio_path}")
            else:
                failures += 1
        except Exception as exc:
            print(f"Transcription failed: {exc}")
            failures += 1

    print(f"\n{'='*60}")
    print(f"Done. {successes} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


def run_single_file_workflow(raw_path: str | None, args: argparse.Namespace) -> int:
    if raw_path:
        path = Path(raw_path.strip().strip('"').strip("'")).expanduser()
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        if not path.is_file():
            print(f"Not a file: {path}")
            return 1
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"Unsupported extension: {path.suffix}. Allowed: {allowed}")
            return 1
        audio_path = path
    else:
        audio_path = prompt_for_audio_file()

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{audio_path.stem}.txt"
    srt_path = (output_dir / f"{audio_path.stem}.srt") if args.srt else None

    print()
    print(f"Loading model: {args.model}")
    print(f"Device: {device} | Compute type: {compute_type}")

    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    try:
        ok = transcribe_file(audio_path, txt_path, srt_path, model, args)
        if ok and args.cleanup:
            audio_path.unlink()
            print(f"Deleted: {audio_path}")
        return 0 if ok else 1
    except Exception as exc:
        print(f"\nTranscription failed: {exc}")
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Track whether --model was explicitly supplied so the menu doesn't override it.
    explicit_model = "--model" in sys.argv

    try:
        if not args.inputs:
            if not explicit_model:
                args.model = prompt_model_menu()
            mode = prompt_mode_menu()
            cleanup_answer = input("\nDelete audio files after transcription? [y/N] ").strip().lower()
            args.cleanup = cleanup_answer == "y"
            if mode == "urls":
                urls = prompt_for_urls()
                return run_project_workflow(urls, args)
            if mode == "batch":
                paths = prompt_for_batch_files()
                return run_multi_file_workflow(paths, args)
            return run_single_file_workflow(None, args)

        urls = [u for u in args.inputs if is_url(u)]
        paths = [p for p in args.inputs if not is_url(p)]

        if urls and paths:
            print("Error: mix of URLs and file paths is not supported. Pass only URLs or only file paths.")
            return 1

        if urls:
            return run_project_workflow(urls, args)

        if len(paths) == 1:
            return run_single_file_workflow(paths[0], args)

        return run_multi_file_workflow(paths, args)

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
