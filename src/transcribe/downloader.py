import logging
import time
from pathlib import Path

from .progress import draw_bar, finish_bar, fmt_eta
from .utils import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_DOWNLOAD_ATTEMPTS = 3


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


def download_audio(
    url: str,
    project_dir: Path,
    stem: str,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> Path:
    """Download audio from a URL using yt-dlp. Returns the path to the extracted audio file."""
    import yt_dlp

    outtmpl = str(project_dir / f"{stem}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
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
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    logger.info("\nDownloading: %s", url)
    try:
        # YouTube intermittently 403s a signed media URL and yt-dlp doesn't retry 4xx.
        # A fresh YoutubeDL re-extracts and gets a new URL (and resumes the .part file).
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                break
            except yt_dlp.utils.DownloadError as exc:
                if "403" not in str(exc) or attempt == _DOWNLOAD_ATTEMPTS:
                    raise
                finish_bar()
                logger.info(
                    "HTTP 403 from host, retrying (%d/%d)...", attempt + 1, _DOWNLOAD_ATTEMPTS
                )
    except Exception as exc:  # yt_dlp may raise DownloadError etc.
        if "ffmpeg" in str(exc).lower() or "ffprobe" in str(exc).lower():
            raise RuntimeError(
                "ffmpeg/ffprobe not found or failed. "
                "URL mode requires `sudo apt install ffmpeg` (or your distro equivalent)."
            ) from exc
        if "403" in str(exc):
            raise RuntimeError(
                f"{exc}\nPersistent 403s usually mean yt-dlp is out of date: "
                "run `uv tool upgrade transcribe`."
            ) from exc
        raise
    finally:
        finish_bar()

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
