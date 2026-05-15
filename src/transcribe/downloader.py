import logging
from pathlib import Path

from .utils import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


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
        "quiet": False,
        "no_warnings": False,
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
