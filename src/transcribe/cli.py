import argparse
from pathlib import Path

from . import config as cfg
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
from .utils import is_url, resolve_cuda_libs


def build_parser(defaults: dict) -> argparse.ArgumentParser:
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
    # --model default is None (sentinel) so we can tell whether the user supplied one.
    # Config/env-provided values are applied in main() after parsing.
    parser.add_argument(
        "--model",
        default=None,
        help=f"Whisper model (e.g. turbo, small, medium, large-v3). Default: {defaults['model']}",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=defaults["device"],
        help=f"Device to use. Default: {defaults['device']}",
    )
    parser.add_argument(
        "--compute-type",
        default=defaults["compute_type"],
        help="Override compute type. Examples: float16, int8_float16, int8.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=defaults["beam_size"],
        help=f"Beam size. Default: {defaults['beam_size']}",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        default=not defaults["vad"],
        help="Disable silence filtering (VAD).",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        default=defaults["srt"],
        help="Also write an .srt subtitle file.",
    )
    parser.add_argument(
        "--language",
        default=defaults["language"],
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
        default=defaults["cleanup"],
        help="Delete audio files after successful transcription.",
    )
    parser.add_argument(
        "--project-root",
        default=str(defaults["project_root"]),
        help=f"Root folder for URL-project subfolders. Default: {defaults['project_root']}",
    )
    return parser


def _classify_inputs(inputs: list[str]) -> tuple[list[str], list[str], bool]:
    """Return (urls, paths, mixed). mixed=True if both kinds are present."""
    urls = [u for u in inputs if is_url(u)]
    paths = [p for p in inputs if not is_url(p)]
    return urls, paths, bool(urls and paths)


def main() -> int:
    resolve_cuda_libs()

    file_cfg = cfg.load_file(cfg.default_config_path())
    env_cfg = cfg.load_env()
    merged = cfg.merge(file_cfg, env_cfg)

    parser = build_parser(merged)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser()

    # Resolve --model sentinel: CLI > env > config > interactive menu (only when no inputs) > builtin.
    cli_supplied_model = args.model is not None
    config_supplied_model = "model" in env_cfg or "model" in file_cfg

    try:
        if not args.inputs:
            if not cli_supplied_model and not config_supplied_model:
                args.model = prompt_model_menu()
            elif not cli_supplied_model:
                args.model = merged["model"]
            mode = prompt_mode_menu()
            default_cleanup = merged["cleanup"]
            prompt_hint = "Y/n" if default_cleanup else "y/N"
            answer = input(f"\nDelete audio files after transcription? [{prompt_hint}] ").strip().lower()
            args.cleanup = default_cleanup if not answer else answer == "y"
            if mode == "urls":
                urls = prompt_for_urls()
                return run_project_workflow(urls, args, project_root)
            if mode == "batch":
                paths = prompt_for_batch_files()
                return run_multi_file_workflow(paths, args)
            return run_single_file_workflow(None, args)

        if not cli_supplied_model:
            args.model = merged["model"]

        urls, paths, mixed = _classify_inputs(args.inputs)
        if mixed:
            print("Error: mix of URLs and file paths is not supported. Pass only URLs or only file paths.")
            return 1

        if urls:
            return run_project_workflow(urls, args, project_root)

        if len(paths) == 1:
            return run_single_file_workflow(paths[0], args)

        return run_multi_file_workflow(paths, args)

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
