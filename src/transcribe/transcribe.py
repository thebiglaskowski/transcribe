import argparse
import logging
import time
from pathlib import Path

from faster_whisper import WhisperModel

from .downloader import download_audio
from .output import write_json, write_srt, write_txt
from .progress import (
    draw_bar,
    draw_two_bars,
    finish_bar,
    fmt_eta,
    reset_two_bars,
    spinner_start,
    spinner_stop,
)
from .prompts import prompt_for_audio_file, prompt_project_name
from .utils import SUPPORTED_EXTENSIONS, default_compute_type, default_device, ffmpeg_available

logger = logging.getLogger(__name__)


def transcribe_file(
    audio_path: Path,
    txt_path: Path,
    srt_path: Path | None,
    json_path: Path | None,
    model: WhisperModel,
    args: argparse.Namespace,
    info_prefix: str = "",
    file_progress: tuple[int, int] | None = None,
) -> bool:
    """Transcribe a single audio file. Returns True on success."""
    logger.info("\n%sInput:  %s", info_prefix, audio_path)
    logger.info("%sOutput: %s", info_prefix, txt_path)
    if srt_path:
        logger.info("%sSRT:    %s", info_prefix, srt_path)

    start_time = time.time()
    use_vad = not args.no_vad

    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=args.beam_size,
        vad_filter=use_vad,
        language=args.language,
        word_timestamps=getattr(args, "word_timestamps", False),
    )

    segments: list = []
    try:
        for segment in segments_iter:
            segments.append(segment)
            if info.duration:
                pct = min(100.0, segment.end / info.duration * 100)
                elapsed = time.time() - start_time
                remaining = elapsed / max(pct / 100, 0.001) - elapsed
                eta = fmt_eta(remaining)
            else:
                pct = 0.0
                eta = ""
            if file_progress is not None:
                draw_two_bars(pct, eta, *file_progress)
            else:
                draw_bar("Transcribing", pct, eta)
    finally:
        if file_progress is not None:
            reset_two_bars()
        else:
            finish_bar()

    if not segments:
        logger.warning("No transcript text was produced.")
        return False

    write_txt(segments, txt_path)
    if srt_path:
        write_srt(segments, srt_path)
    if json_path:
        write_json(segments, info, json_path)

    elapsed = time.time() - start_time
    logger.info("Language: %s (%.2f) | %.1fs", info.language, info.language_probability, elapsed)
    logger.info("Saved: %s", txt_path)
    return True


def run_project_workflow(urls: list[str], args: argparse.Namespace, project_root: Path) -> int:
    project_name, project_dir = prompt_project_name(project_root)

    if not ffmpeg_available():
        logger.error(
            "ffmpeg not found in PATH. URL/project mode requires it for audio extraction.\n"
            "Install with: sudo apt install ffmpeg"
        )
        return 1

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)

    logger.info("\nProject folder: %s", project_dir)
    logger.info("Videos to process: %d", len(urls))

    _spinner = spinner_start(f"Loading model {args.model}…")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    finally:
        spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)

    successes = 0
    failures = 0
    bar = "=" * 60

    for i, url in enumerate(urls, start=1):
        stem = f"{project_name}{i}"
        logger.info("\n%s", bar)
        logger.info("[%d/%d] %s", i, len(urls), url)
        logger.info("%s", bar)

        txt_path = project_dir / f"{stem}.txt"
        srt_path = (project_dir / f"{stem}.srt") if args.srt else None
        json_path = None
        if getattr(args, "json", False):
            json_path = project_dir / f"{stem}.json"

        if txt_path.exists() and txt_path.stat().st_size > 0:
            # Also consider json if --json was requested (simple resume)
            if not json_path or (json_path.exists() and json_path.stat().st_size > 0):
                logger.info("  Skipping — transcript already exists: %s", txt_path.name)
                successes += 1
                continue

        try:
            audio_path = download_audio(
                url,
                project_dir,
                stem,
                cookies_from_browser=getattr(args, "cookies_from_browser", None),
                cookies_file=getattr(args, "cookies", None),
            )
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            if "ffmpeg" in str(exc).lower():
                logger.error("Hint: install ffmpeg (sudo apt install ffmpeg)")
            failures += 1
            continue

        try:
            ok = transcribe_file(
                audio_path,
                txt_path,
                srt_path,
                json_path,
                model,
                args,
                info_prefix="  ",
                file_progress=(i, len(urls)),
            )
            if ok:
                successes += 1
                if args.cleanup:
                    audio_path.unlink()
                    logger.info("Deleted: %s", audio_path)
            else:
                failures += 1
        except Exception as exc:
            msg = str(exc)
            hint = ""
            if "cuda" in msg.lower():
                hint = " (try --device cpu)"
            logger.error("Transcription failed: %s%s", exc, hint)
            failures += 1

    logger.info("\n%s", bar)
    logger.info("Done. %d succeeded, %d failed.", successes, failures)
    logger.info("Transcripts saved in: %s", project_dir)
    return 0 if failures == 0 else 1


def run_multi_file_workflow(raw_paths: list[str], args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for raw in raw_paths:
        p = Path(raw.strip().strip('"').strip("'")).expanduser()
        if not p.exists():
            logger.error("File not found: %s", p)
            return 1
        if not p.is_file():
            logger.error("Not a file: %s", p)
            return 1
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            logger.error("Unsupported extension: %s. Allowed: %s", p.suffix, allowed)
            return 1
        paths.append(p)

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)

    logger.info("\nFiles to process: %d", len(paths))

    _spinner = spinner_start(f"Loading model {args.model}…")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    finally:
        spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)

    output_dir_override = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir_override:
        output_dir_override.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    bar = "=" * 60

    for i, audio_path in enumerate(paths, start=1):
        logger.info("\n%s", bar)
        logger.info("[%d/%d] %s", i, len(paths), audio_path.name)
        logger.info("%s", bar)
        output_dir = output_dir_override or audio_path.parent
        txt_path = output_dir / f"{audio_path.stem}.txt"
        srt_path = (output_dir / f"{audio_path.stem}.srt") if args.srt else None
        json_path = None
        if getattr(args, "json", False):
            json_path = output_dir / f"{audio_path.stem}.json"

        if txt_path.exists() and txt_path.stat().st_size > 0:
            if not json_path or (json_path.exists() and json_path.stat().st_size > 0):
                logger.info("  Skipping — transcript already exists: %s", txt_path.name)
                successes += 1
                continue

        try:
            ok = transcribe_file(
                audio_path,
                txt_path,
                srt_path,
                json_path,
                model,
                args,
                info_prefix="  ",
                file_progress=(i, len(paths)),
            )
            if ok:
                successes += 1
                if args.cleanup:
                    audio_path.unlink()
                    logger.info("Deleted: %s", audio_path)
            else:
                failures += 1
        except Exception as exc:
            msg = str(exc)
            hint = ""
            if "cuda" in msg.lower():
                hint = " (try --device cpu)"
            logger.error("Transcription failed: %s%s", exc, hint)
            failures += 1

    logger.info("\n%s", bar)
    logger.info("Done. %d succeeded, %d failed.", successes, failures)
    return 0 if failures == 0 else 1


def run_single_file_workflow(raw_path: str | None, args: argparse.Namespace) -> int:
    if raw_path:
        path = Path(raw_path.strip().strip('"').strip("'")).expanduser()
        if not path.exists():
            logger.error("File not found: %s", path)
            return 1
        if not path.is_file():
            logger.error("Not a file: %s", path)
            return 1
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            logger.error("Unsupported extension: %s. Allowed: %s", path.suffix, allowed)
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
    json_path = (output_dir / f"{audio_path.stem}.json") if getattr(args, "json", False) else None

    logger.info("")
    _spinner = spinner_start(f"Loading model {args.model}…")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    finally:
        spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)

    try:
        ok = transcribe_file(audio_path, txt_path, srt_path, json_path, model, args)
        if ok and args.cleanup:
            audio_path.unlink()
            logger.info("Deleted: %s", audio_path)
        return 0 if ok else 1
    except Exception as exc:
        msg = str(exc)
        hint = ""
        if "cuda" in msg.lower() or "cublas" in msg.lower() or "cudnn" in msg.lower():
            hint = " (try --device cpu or install the [cuda] extra + NVIDIA drivers)"
        logger.error("\nTranscription failed: %s%s", exc, hint)
        return 1
