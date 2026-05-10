from __future__ import annotations

import argparse
import ctypes
import os
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
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD = True
DEFAULT_SRT = False


def cuda_available() -> bool:
    """Best-effort check for the CUDA runtime libraries faster-whisper needs."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio/video file with faster-whisper and save a .txt transcript.",
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to the input audio/video file. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model to use. Examples: turbo, small, medium, large-v3. Default: turbo",
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
        help="Override compute type. Examples: float16, int8_float16, int8. Default: auto based on device",
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
        help="Also write an .srt subtitle file next to the .txt transcript.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Source language code (e.g. en, es, fr). Default: auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to the input file's directory.",
    )
    return parser


def resolve_input_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path.strip().strip('"').strip("'")).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported extension: {path.suffix}. Allowed: {allowed}")
        return path
    return prompt_for_audio_file()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        audio_path = resolve_input_path(args.audio_file)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)
    use_vad = not args.no_vad

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{audio_path.stem}.txt"
    srt_path = output_dir / f"{audio_path.stem}.srt"

    print()
    print(f"Loading model: {args.model}")
    print(f"Device: {device} | Compute type: {compute_type}")
    print(f"Input:  {audio_path}")
    print(f"Output: {txt_path}")
    if args.srt:
        print(f"SRT:    {srt_path}")
    print()

    start_time = time.time()

    try:
        print(f"(Model will be downloaded on first use if not cached.)")
        model = WhisperModel(
            args.model,
            device=device,
            compute_type=compute_type,
        )

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
            return 1

        write_txt(segments, txt_path)
        if args.srt:
            write_srt(segments, srt_path)

        elapsed = time.time() - start_time
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"Saved transcript to: {txt_path}")
        if args.srt:
            print(f"Saved subtitles to: {srt_path}")
        print(f"Completed in {elapsed:.1f} seconds")
        return 0

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"\nTranscription failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
