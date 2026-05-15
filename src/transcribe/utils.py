import ctypes
import re

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


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def sanitize_project_name(raw: str) -> str | None:
    """Return a filesystem-safe project name, or None if no usable characters remain."""
    stripped = raw.strip()
    if not stripped:
        return None
    safe = re.sub(r"[^\w\-]", "-", stripped).strip("-")
    return safe or None


def cuda_available() -> bool:
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
