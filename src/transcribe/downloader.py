import logging
import time
from pathlib import Path

from .progress import draw_bar, finish_bar, fmt_eta
from .utils import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def _make_progress_hook():
    start = [time.time()]

    def hook(d: dict) -> None:
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if not total:
                return
            pct = d["downloaded_bytes"] / total * 100
            elapsed = time.time() - start[0]
            remaining = elapsed / max(pct / 100, 0.001) - elapsed
            draw_bar("Downloading", pct, fmt_eta(remaining))
        elif d["status"] == "finished":
            finish_bar()

    return hook


def download_audio(url: str, project_dir: Path, stem: str) -> Path:
    """Download audio from a URL using yt-dlp. Returns the path to the extracted audio file."""
    import yt_dlp

    outtmpl = str(project_dir / f"{stem}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=opus]/bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "keepvideo": False,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_make_progress_hook()],
    }

    logger.info("\nDownloading: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = project_dir / f"{stem}.wav"
    if not audio_path.exists():
        candidates = list(project_dir.glob(f"{stem}.*"))
        audio_candidates = [p for p in candidates if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not audio_candidates:
            raise FileNotFoundError(
                f"Could not find downloaded audio for stem '{stem}' in {project_dir}"
            )
        audio_path = audio_candidates[0]

    return audio_path
