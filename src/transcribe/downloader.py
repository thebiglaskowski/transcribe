import glob
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


def _cookie_opts(cookies_from_browser: str | None, cookies_file: str | None) -> dict:
    opts: dict = {}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    if cookies_file:
        opts["cookiefile"] = cookies_file
    return opts


def list_sources(
    url: str,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> list[tuple[str | None, list[dict]]]:
    """Expand a URL into the videos it points at, without downloading anything.

    Returns [(title, videos), ...] with videos as [{"id", "title", "url"}, ...]:
    - a single video → [(None, [itself])]
    - a playlist or TikTok profile → [(its title, entries)], newest first for profiles
    - a YouTube channel root → one group per non-empty tab (Videos, Live, Shorts…)
    """
    import yt_dlp

    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        **_cookie_opts(cookies_from_browser, cookies_file),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    def video(e: dict) -> dict:
        # flat entries carry the watch page in "url"; a full single-video info in "webpage_url"
        return {"id": e["id"], "title": e.get("title"), "url": e.get("webpage_url") or e["url"]}

    def flatten(entries) -> list[dict]:
        out: list[dict] = []
        for e in entries:
            if e:
                out.extend(flatten(e["entries"]) if "entries" in e else [video(e)])
        return out

    if "entries" not in info:
        return [(None, [video(info)])]
    entries = [e for e in info["entries"] if e]
    if entries and all("entries" in e for e in entries):  # channel root: tabs
        groups = [(e.get("title"), flatten(e["entries"])) for e in entries]
        return [(title, videos) for title, videos in groups if videos]
    return [(info.get("title"), flatten(entries))]


def download_audio(
    url: str,
    project_dir: Path,
    stem: str,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> tuple[Path, dict]:
    """Download audio from a URL using yt-dlp.

    Returns (path to the extracted audio file, yt-dlp's info dict for the video).
    """
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
        "noprogress": True,  # quiet doesn't imply it in the API; our hook draws the bar
        "progress_hooks": [_make_progress_hook()],
        # a short pause before each download keeps channel-sized runs under YouTube's
        # bot-detection radar; negligible next to transcription time
        "sleep_interval": 2,
        "max_sleep_interval": 6,
        **_cookie_opts(cookies_from_browser, cookies_file),
    }

    logger.info("\nDownloading: %s", url)
    try:
        # YouTube intermittently 403s a signed media URL and yt-dlp doesn't retry 4xx.
        # A fresh YoutubeDL re-extracts and gets a new URL (and resumes the .part file).
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
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
        candidates = list(project_dir.glob(f"{glob.escape(stem)}.*"))  # stems contain [id]
        audio_candidates = [p for p in candidates if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not audio_candidates:
            raise FileNotFoundError(
                f"Could not find downloaded audio for stem '{stem}' in {project_dir}"
            )
        audio_path = audio_candidates[0]

    return audio_path, info
