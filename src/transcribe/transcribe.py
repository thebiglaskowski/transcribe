import argparse
import bisect
import logging
import os
import re
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from rich.text import Text

from .downloader import download_audio, list_sources
from .output import source_header, write_json, write_srt, write_txt
from .progress import display, fmt_duration, status, tracker
from .prompts import (
    prompt_for_audio_file,
    prompt_pick_sources,
    prompt_project_name,
    prompt_video_count,
)
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
    with status(f"🧠 Loading {args.model} on {device}…"):
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    diarizer = None
    if getattr(args, "diarize", False):
        with status("👥 Loading the speaker model…"):
            diarizer = load_diarizer(device)
    logger.info(
        Text.assemble(
            "\n🎧 ",
            (args.model, "bold cyan"),
            (f" · {device} {compute_type}", "dim"),
            (" · 👥 speaker labels", "dim") if diarizer else "",
        )
    )
    return model, diarizer


def _done_line(title: str, summary: dict) -> Text:
    """One finished-item line: ✅ Title  12:31 · 👥 2 · ⚡ 0:38"""
    if summary["no_speech"]:
        return Text.assemble("🔇 ", (display(title), "bold"), ("  no speech", "dim"))
    bits = [fmt_duration(summary["duration"])]
    if summary["speakers"] is not None:
        bits.append(f"👥 {summary['speakers']}")
    bits.append(f"⚡ {fmt_duration(summary['elapsed'])}")
    return Text.assemble("✅ ", (display(title), "bold"), ("  " + " · ".join(bits), "dim"))


def _summary_line(successes: int, failures: int, started: float) -> Text:
    return Text.assemble(
        "\n🎉 " if not failures else "\n🏁 ",
        ("Done", "bold green" if not failures else "bold yellow"),
        f" — {successes} transcribed",
        (f" · {failures} failed", "bold red") if failures else "",
        (f" · {fmt_duration(time.time() - started)}", "dim"),
    )


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
    diarizer=None,
    header: str | None = None,
    report=lambda stage, pct=None: None,
) -> dict:
    """Transcribe one audio file and write its outputs.

    `report(stage_emoji, pct)` receives progress (pct=None: working, no percentage).
    Returns {"duration", "speakers" (None unless diarized), "no_speech", "elapsed"}.
    """
    logger.debug("Input: %s → %s", audio_path, txt_path)
    start_time = time.time()
    use_vad = not args.no_vad

    report("📝")
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
    for segment in segments_iter:
        segments.append(segment)
        if info.duration:
            report("📝", min(100.0, segment.end / info.duration * 100))

    no_speech = not segments
    if no_speech:
        # A placeholder (not a missing file) so resume skips music-only videos instead of
        # re-downloading them on every channel re-sync.
        segments = [SimpleNamespace(start=0.0, end=0.0, text="[no speech detected]", words=[])]
        diarizer = None

    speakers = None
    if diarizer:
        report("👥")
        segments, speakers = split_by_speaker(segments, speaker_turns(diarizer, audio))

    write_txt(segments, txt_path, speakers, header)
    if srt_path:
        write_srt(segments, srt_path, speakers)
    if json_path:
        write_json(segments, info, json_path, speakers)

    logger.debug("Language: %s (%.2f)", info.language, info.language_probability)
    return {
        "duration": info.duration,
        "speakers": len({s for s in speakers if s}) if speakers is not None else None,
        "no_speech": no_speech,
        "elapsed": time.time() - start_time,
    }


# Known failure → what to do about it. Matched against the lowercased error text.
_ERROR_HINTS = [
    ("confirm your age", "🔞 age-restricted: needs cookies from a browser signed in to YouTube"),
    ("not a bot", "🤖 YouTube bot check: sign in via cookies, or pause and re-run later"),
    ("403", "if this keeps happening: uv tool upgrade transcribe"),
    ("cudnn", "check the [cuda] extra, or try --device cpu"),
    ("cublas", "check the [cuda] extra, or try --device cpu"),
    ("cuda", "check the [cuda] extra, or try --device cpu"),
]


def short_error(exc: Exception) -> str:
    """One line for the terminal: the error's first sentence (minus yt-dlp's
    "ERROR: [youtube] <id>: " preamble) plus a hint when we know the fix. -v logs it in full."""
    # yt-dlp colors "ERROR:" when it sees a terminal; drop escape codes before matching
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip() or type(exc).__name__
    first = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*[\w-]+:\s*)?", "", text.splitlines()[0])
    first = first.split(". ")[0].rstrip(".")
    lowered = text.lower()
    hint = next((h for key, h in _ERROR_HINTS if key in lowered), None)
    return f"{first} · {hint}" if hint else first


def _plan_sources(urls: list[str], cookies: dict) -> tuple[list[tuple[str | None, list]], int]:
    """List every URL and ask all the questions up front, so long runs go unattended.

    Channel tabs (Videos/Live/Shorts), playlists and TikTok profiles each become their own
    source with a "how many" prompt; plain video URLs share one untitled source, listed first.
    Returns ([(source title or None, videos)], number of URLs that couldn't be read).
    """
    singles: list[dict] = []
    sources: list[tuple[str | None, list]] = []
    failed = 0
    for url in urls:
        try:
            with status(f"🔎 Looking up {url}…"):  # closed before any prompt below
                groups = list_sources(url, **cookies)
        except Exception as exc:
            groups = exc
        if isinstance(groups, Exception):
            logger.debug("Lookup error in full: %s", groups)
            logger.error("Could not read %s: %s", url, short_error(groups))
            failed += 1
            continue
        if groups[0][0] is None:
            singles.extend(groups[0][1])
            continue
        if len(groups) > 1:
            groups = prompt_pick_sources(url, groups)
        for title, videos in groups:
            count = prompt_video_count(title or url, len(videos))
            if count:
                sources.append((title, videos[:count]))
    return ([(None, singles)] if singles else []) + sources, failed


def _outputs(project_dir: Path, stem: str, args: argparse.Namespace):
    txt = project_dir / f"{stem}.txt"
    srt = (project_dir / f"{stem}.srt") if args.srt else None
    js = (project_dir / f"{stem}.json") if getattr(args, "json", False) else None
    return txt, srt, js


def _is_done(project_dir: Path, stem: str, args: argparse.Namespace) -> bool:
    txt, _, js = _outputs(project_dir, stem, args)
    return all(p is None or (p.exists() and p.stat().st_size > 0) for p in (txt, js))


def _process_project(
    project_dir: Path, pending: list[dict], args: argparse.Namespace, model, diarizer, cookies
) -> tuple[int, int]:
    """Download + transcribe each pending video into project_dir. Returns (succeeded, failed)."""
    successes = failures = 0
    with tracker(project_dir.name, len(pending)) as t:
        for v in pending:
            title = v["title"] or v["url"]
            t.start(title)
            txt_path, srt_path, json_path = _outputs(project_dir, v["stem"], args)
            try:
                audio_path, info = download_audio(
                    v["url"],
                    project_dir,
                    v["stem"],
                    **cookies,
                    on_progress=lambda pct: t.stage("📥", pct),
                )
            except Exception as exc:
                logger.debug("Download error in full: %s", exc)
                logger.error("%s — download failed: %s", display(title), short_error(exc))
                failures += 1
                t.advance()
                continue
            try:
                summary = transcribe_file(
                    audio_path,
                    txt_path,
                    srt_path,
                    json_path,
                    model,
                    args,
                    diarizer=diarizer,
                    header=source_header(info),
                    report=t.stage,
                )
                logger.info(_done_line(title, summary))
                successes += 1
                if args.cleanup:
                    audio_path.unlink()
                    logger.debug("Deleted: %s", audio_path)
            except Exception as exc:
                logger.debug("Transcription error in full: %s", exc, exc_info=True)
                logger.error("%s — transcription failed: %s", display(title), short_error(exc))
                failures += 1
            t.advance()
    return successes, failures


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
    sources, failures = _plan_sources(urls, cookies)
    if not sources:
        logger.error("Nothing to transcribe.")
        return 1 if failures else 0

    # One project folder per source. Resume: a video counts as done when its transcript (and
    # json, if requested) exists; names carry the video id, so this holds as a channel's
    # newest-first order shifts.
    # ponytail: keyed on the full "<title> [id]" name — a video renamed upstream gets redone.
    projects: list[tuple[Path, list[dict]]] = []
    successes = 0
    for title, videos in sources:
        default = sanitize_project_name(title.replace(" - ", "-")) if title else None
        _, project_dir = prompt_project_name(project_root, default=default)
        seen: set[str] = set()
        videos = [v for v in videos if not (v["id"] in seen or seen.add(v["id"]))]
        for v in videos:
            v["stem"] = video_stem(v["title"], v["id"])
        pending = [v for v in videos if not _is_done(project_dir, v["stem"], args)]
        successes += len(videos) - len(pending)
        projects.append((project_dir, pending))
        already = len(videos) - len(pending)
        logger.info(
            Text.assemble(
                "📁 ",
                (project_dir.name, "bold magenta"),
                f"  {len(pending)} to transcribe",
                (f" · {already} already done", "dim") if already else "",
            )
        )

    if not any(pending for _, pending in projects):
        return 0 if failures == 0 else 1

    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)
    started = time.time()
    model, diarizer = _load_models(args, device, compute_type)

    for project_dir, pending in projects:
        if not pending:
            continue
        logger.info(
            Text.assemble(
                "\n📺 ", (project_dir.name, "bold magenta"), (f"  {len(pending)} videos", "dim")
            )
        )
        ok, failed = _process_project(project_dir, pending, args, model, diarizer, cookies)
        successes += ok
        failures += failed

    logger.info(_summary_line(successes, failures, started))
    for project_dir, _ in projects:
        logger.info(Text.assemble("📁 ", (str(project_dir), "dim")))
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

    output_dir_override = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir_override:
        output_dir_override.mkdir(parents=True, exist_ok=True)

    def outputs(path: Path):
        out = output_dir_override or path.parent
        return _outputs(out, path.stem, args)

    pending = [p for p in paths if not _is_done(outputs(p)[0].parent, p.stem, args)]
    already = len(paths) - len(pending)
    logger.info(
        Text.assemble(
            "\n📂 ",
            f"{len(pending)} files to transcribe",
            (f" · {already} already done", "dim") if already else "",
        )
    )
    if not pending:
        return 0

    started = time.time()
    device = default_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)
    model, diarizer = _load_models(args, device, compute_type)

    successes, failures = already, 0
    with tracker("Files", len(pending)) as t:
        for audio_path in pending:
            t.start(audio_path.name)
            try:
                summary = transcribe_file(
                    audio_path, *outputs(audio_path), model, args, diarizer=diarizer, report=t.stage
                )
                logger.info(_done_line(audio_path.name, summary))
                successes += 1
                if args.cleanup:
                    audio_path.unlink()
                    logger.debug("Deleted: %s", audio_path)
            except Exception as exc:
                logger.debug("Transcription error in full: %s", exc, exc_info=True)
                logger.error("%s — transcription failed: %s", audio_path.name, short_error(exc))
                failures += 1
            t.advance()

    logger.info(_summary_line(successes, failures, started))
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

    model, diarizer = _load_models(args, device, compute_type)

    try:
        with tracker() as t:
            t.start(audio_path.name)
            summary = transcribe_file(
                audio_path,
                txt_path,
                srt_path,
                json_path,
                model,
                args,
                diarizer=diarizer,
                report=t.stage,
            )
    except Exception as exc:
        logger.debug("Transcription error in full: %s", exc, exc_info=True)
        logger.error("Transcription failed: %s", short_error(exc))
        return 1
    logger.info(_done_line(audio_path.name, summary))
    logger.info(Text.assemble("📄 ", (str(txt_path), "dim")))
    if args.cleanup:
        audio_path.unlink()
        logger.debug("Deleted: %s", audio_path)
    return 0
