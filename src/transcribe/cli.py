import argparse
from pathlib import Path

from .prompts import (
    prompt_for_batch_files,
    prompt_for_urls,
    prompt_mode_menu,
    prompt_model_menu,
)
from .transcribe import (
    run_multi_file_workflow,
    run_project_workflow,
    run_single_file_workflow,
)
from .utils import is_url

DEFAULT_MODEL = "turbo"
DEFAULT_BEAM_SIZE = 5
DEFAULT_PROJECT_ROOT = Path.home() / ".transcribe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
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
        default=None,
        help=f"Whisper model (e.g. turbo, small, medium, large-v3). Default: {DEFAULT_MODEL}",
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
        default=False,
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
        help="Output directory for single/batch mode. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Delete audio files after successful transcription.",
    )
    return parser


def _classify_inputs(inputs: list[str]) -> tuple[list[str], list[str], bool]:
    """Return (urls, paths, mixed). mixed=True if both kinds are present."""
    urls = [u for u in inputs if is_url(u)]
    paths = [p for p in inputs if not is_url(p)]
    return urls, paths, bool(urls and paths)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if not args.inputs:
            if args.model is None:
                args.model = prompt_model_menu()
            mode = prompt_mode_menu()
            cleanup_answer = input("\nDelete audio files after transcription? [y/N] ").strip().lower()
            args.cleanup = cleanup_answer == "y"
            if mode == "urls":
                urls = prompt_for_urls()
                return run_project_workflow(urls, args, DEFAULT_PROJECT_ROOT)
            if mode == "batch":
                paths = prompt_for_batch_files()
                return run_multi_file_workflow(paths, args)
            return run_single_file_workflow(None, args)

        if args.model is None:
            args.model = DEFAULT_MODEL

        urls, paths, mixed = _classify_inputs(args.inputs)
        if mixed:
            print("Error: mix of URLs and file paths is not supported. Pass only URLs or only file paths.")
            return 1

        if urls:
            return run_project_workflow(urls, args, DEFAULT_PROJECT_ROOT)

        if len(paths) == 1:
            return run_single_file_workflow(paths[0], args)

        return run_multi_file_workflow(paths, args)

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
