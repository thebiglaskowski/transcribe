import ctypes
import os
import re
from pathlib import Path

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


def resolve_cuda_libs() -> None:
    """Preload libcublas and libcudnn from the installed nvidia-cu12 wheels.

    Why: faster-whisper/ctranslate2 dlopen these by their soname, which requires the
    nvidia wheel directories to be on the dynamic linker's search path. Rather than
    asking users to set LD_LIBRARY_PATH, we look up the wheel paths from the running
    interpreter and preload the libs with RTLD_GLOBAL so subsequent dlopen calls
    resolve against them. No-op if the nvidia wheels are not installed.
    """
    try:
        import nvidia.cublas.lib as cublas_mod
        import nvidia.cudnn.lib as cudnn_mod
    except ImportError:
        return

    cublas_dir = Path(cublas_mod.__file__).parent
    cudnn_dir = Path(cudnn_mod.__file__).parent

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(cublas_dir), str(cudnn_dir)]
    if current:
        parts.append(current)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    for libname, libdir in (("libcublas.so.12", cublas_dir), ("libcudnn.so.9", cudnn_dir)):
        lib_path = libdir / libname
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


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
