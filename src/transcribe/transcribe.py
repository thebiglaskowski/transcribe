import argparse
import bisect
import logging
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

from .downloader import download_audio, list_videos
from .output import source_header, write_json, write_srt, write_txt
from .progress import (
    draw_bar,
    draw_two_bars,
    finish_bar,
    fmt_eta,
    reset_two_bars,
    spinner_start,
    spinner_stop,
)
from .prompts import prompt_for_audio_file, prompt_project_name, prompt_video_count
from .utils import (
    SUPPORTED_EXTENSIONS,
    default_compute_type,
    default_device,
    ffmpeg_available,
    sanitize_project_name,
    video_stem,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 8  # measured: ~1.9x faster than sequential on a 3060 Ti; 16 was no faster
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def load_diarizer(device: str):
    """Load the pyannote speaker-diarization pipeline (lazy: torch is heavy)."""
    # cuDNN probes a ~10 GB workspace on first use, fails, and falls back; torch logs each
    # attempt as an OOM warning. Harmless (same result at any batch size), so hide C++ warnings.
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker labels need the [diarize] extra: "
            'uv tool install --force --reinstall "/path/to/transcribe[cuda,diarize]"'
        ) from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pyannote's TF32/reproducibility notices
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)  # token from `hf auth login`
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {DIARIZATION_MODEL}: {exc}\n"
            f"Accept its terms at https://hf.co/{DIARIZATION_MODEL}, then log in with "
            "`uvx --from huggingface_hub hf auth login` (a Read token is enough)."
        ) from exc
    pipeline.to(torch.device(device))
    return pipeline


def _load_models(args: argparse.Namespace, device: str, compute_type: str):
    _spinner = spinner_start(f"Loading model {args.model}…")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
        diarizer = load_diarizer(device) if getattr(args, "diarize", False) else None
    finally:
        spinner_stop(_spinner)
    logger.info("Device: %s | Compute type: %s", device, compute_type)
    if diarizer:
        logger.info("Speaker labels: on")
    return model, diarizer


def speaker_turns(diarizer, audio) -> list[tuple[float, float, str]]:
    """Run diarization on 16 kHz mono audio; returns (start, end, speaker) turns."""
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # std()-degrees-of-freedom noise on short turns
        result = diarizer({"waveform": torch.from_numpy(audio)[None], "sample_rate": 16000})
    # exclusive_* has no overlapping turns — pyannote's variant meant for aligning to transcripts
    return [(t.start, t.end, spk) for t, spk in result.exclusive_speaker_diarization]


def assign_speakers(items: list, turns: list[tuple[float, float, str]]) -> list[str | None]:
    """Label each item (segment or word) with the speaker who talks most during it.

    Labels are "Speaker 1", "Speaker 2"… numbered by first appearance; None when no turn
    overlaps (music, silence). `turns` must be sorted and non-overlapping, which pyannote's
    exclusive diarization guarantees — so both starts and ends ascend and bisect works.
    """
    starts = [t[0] for t in turns]
    names: dict[str, str] = {}
    labels: list[str | None] = []
    for item in items:
        overlap: dict[str, float] = {}
        i = bisect.bisect_left(starts, item.end) - 1  # last turn starting before item ends
        while i >= 0 and turns[i][1] > item.start:
            start, end, spk = turns[i]
            overlap[spk] = overlap.get(spk, 0.0) + min(item.end, end) - max(item.start, start)
            i -= 1
        if not overlap:
            labels.append(None)
            continue
        spk = max(overlap, key=overlap.__getitem__)
        labels.append(names.setdefault(spk, f"Speaker {len(names) + 1}"))
    return labels


def split_by_speaker(segments: list, turns: list[tuple[float, float, str]]) -> tuple[list, list]:
    """Label words by speaker and split segments where the speaker changes mid-sentence.

    Needs word timestamps. Returns (segments, speakers) as aligned lists for the writers.
    Unlabeled words (no overlapping turn) stay with the words around them.
    """
    labels = iter(assign_speakers([w for seg in segments for w in seg.words or []], turns))
    out: list = []
    speakers: list[str | None] = []

    def flush(words: list, speaker: str | None) -> None:
        text = "".join(w.word for w in words)
        out.append(SimpleNamespace(start=words[0].start, end=words[-1].end, text=text, words=words))
        speakers.append(speaker)

    for seg in segments:
        run: list = []
        run_speaker = None
        for word in seg.words or []:
            speaker = next(labels)
            if run and speaker and run_speaker and speaker != run_speaker:
                flush(run, run_speaker)
                run, run_speaker = [], None
            run.append(word)
            run_speaker = run_speaker or speaker
        if run:
            flush(run, run_speaker)
    return out, speakers


def transcribe_file(
    audio_path: Path,
    txt_path: Path,
    srt_path: Path | None,
    json_path: Path | None,
    model: WhisperModel,
    args: argparse.Namespace,
    info_prefix: str = "",
    file_progress: tuple[int, int] | None = None,
    diarizer=None,
    header: str | None = None,
) -> bool:
    """Transcribe a single audio file. Returns True on success."""
    logger.info("\n%sInput:  %s", info_prefix, audio_path)
    logger.info("%sOutput: %s", info_prefix, txt_path)
    if srt_path:
        logger.info("%sSRT:    %s", info_prefix, srt_path)

    start_time = time.time()
    use_vad = not args.no_vad

    audio = decode_audio(str(audio_path))
    options = dict(
        beam_size=args.beam_size,
        language=args.language,
        # speaker labels are assigned per word, so diarizing needs word timestamps
        word_timestamps=getattr(args, "word_timestamps", False) or diarizer is not None,
    )
    if use_vad:
        # Batched needs VAD to cut the audio into chunks. without_timestamps=False keeps
        # sentence-sized segments (default batched output is one ~30 s segment per chunk).
        segments_iter, info = BatchedInferencePipeline(model).transcribe(
            audio, batch_size=BATCH_SIZE, without_timestamps=False, **options
        )
    else:
        segments_iter, info = model.transcribe(audio, vad_filter=False, **options)

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
        # A placeholder (not a missing file) so resume skips music-only videos instead of
        # re-downloading them on every channel re-sync.
        logger.warning("%sNo speech detected; writing a placeholder.", info_prefix)
        segments = [SimpleNamespace(start=0.0, end=0.0, text="[no speech detected]", words=[])]
        diarizer = None

    speakers = None
    if diarizer:
        _spinner = spinner_start("Identifying speakers…")
        try:
            segments, speakers = split_by_speaker(segments, speaker_turns(diarizer, audio))
        finally:
            spinner_stop(_spinner)
        logger.info("%sSpeakers: %d", info_prefix, len({s for s in speakers if s}))

    write_txt(segments, txt_path, speakers, header)
    if srt_path:
        write_srt(segments, srt_path, speakers)
    if json_path:
        write_json(segments, info, json_path, speakers)

    elapsed = time.time() - start_time
    logger.info("Language: %s (%.2f) | %.1fs", info.language, info.language_probability, elapsed)
    logger.info("Saved: %s", txt_path)
    return True


def _expand_urls(urls: list[str], cookies: dict) -> tuple[list[dict], str | None, int]:
    """List every video behind the URLs (channels/playlists expand), deduped by id.

    Returns (videos, playlist title when exactly one URL was a playlist, URLs that failed).
    """
    videos: list[dict] = []
    titles: list[str] = []
    failed = 0
    _spinner = spinner_start("Looking up videos…")
    try:
        for url in urls:
            try:
                title, found = list_videos(url, **cookies)
            except Exception as exc:
                spinner_stop(_spinner)
                logger.error("Could not read %s: %s", url, exc)
                _spinner = spinner_start("Looking up videos…")
                failed += 1
                continue
            if title:
                titles.append(title)
            videos.extend(found)
    finally:
        spinner_stop(_spinner)
    seen: set[str] = set()
    unique = [v for v in videos if not (v["id"] in seen or seen.add(v["id"]))]
    return unique, (titles[0] if len(urls) == 1 and titles else None), failed


def run_project_workflow(urls: list[str], args: argparse.Namespace, project_root: Path) -> int:
    if not ffmpeg_available():
        logger.error(
            "ffmpeg not found in PATH. URL/project mode requires it for audio extraction.\n"
            "Install with: sudo apt install ffmpeg"
        )
        return 1

    cookies = dict(
        cookies_from_browser=getattr(args, "cookies_from_browser", None),
        cookies_file=getattr(args, "cookies", None),
    )
    videos, playlist_title, failures = _expand_urls(urls, cookies)
    if not videos:
        logger.error("No videos found.")
        return 1
    if len(videos) > len(urls):  # a channel/playlist/profile expanded
        logger.info(
            "\nFound %d videos%s.", len(videos), f" in {playlist_title}" if playlist_title else ""
        )
        count = prompt_video_count(len(videos))
        if count == 0:
            print("Cancelled.")
            return 0
        videos = videos[:count]

    default_name = sanitize_project_name(playlist_title) if playlist_title else None
    _, project_dir = prompt_project_name(project_root, default=default_name)

    def outputs(stem: str) -> tuple[Path, Path | None, Path | None]:
        txt = project_dir / f"{stem}.txt"
        srt = (project_dir / f"{stem}.srt") if args.srt else None
        js = (project_dir / f"{stem}.json") if getattr(args, "json", False) else None
        return txt, srt, js

    def done(stem: str) -> bool:
        txt, _, js = outputs(stem)
        return all(p is None or (p.exists() and p.stat().st_size > 0) for p in (txt, js))

    # Resume: a video counts as done when its transcript (and json, if requested) exists.
    # Names carry the video id, so this holds even as a channel's newest-first order shifts.
    # ponytail: keyed on the full "<title> [id]" name — a video renamed upstream gets redone.
    for v in videos:
        v["stem"] = video_stem(v["title"], v["id"])
    pending = [v for v in videos if not done(v["stem"])]
    already = len(videos) - len(pending)

    logger.info("\nProject folder: %s", project_dir)
    logger.info(
        "Videos to process: %d%s",
        len(pending),
        f" ({already} already transcribed, skipped)" if already else "",
    )
    if not pending:
        return 0 if failures == 0 else 1

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)
    model, diarizer = _load_models(args, device, compute_type)

    successes = already
    bar = "=" * 60

    for i, v in enumerate(pending, start=1):
        stem = v["stem"]
        logger.info("\n%s", bar)
        logger.info("[%d/%d] %s", i, len(pending), v["title"] or v["url"])
        logger.info("%s", bar)
        txt_path, srt_path, json_path = outputs(stem)

        try:
            audio_path, info = download_audio(v["url"], project_dir, stem, **cookies)
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
                file_progress=(i, len(pending)),
                diarizer=diarizer,
                header=source_header(info),
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

    model, diarizer = _load_models(args, device, compute_type)

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
                diarizer=diarizer,
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
    model, diarizer = _load_models(args, device, compute_type)

    try:
        ok = transcribe_file(
            audio_path, txt_path, srt_path, json_path, model, args, diarizer=diarizer
        )
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
