"""Layered configuration: builtin defaults < TOML file < environment < CLI flags."""

import os
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Settings:
    model: str = "turbo"
    device: str = "auto"
    compute_type: str | None = None
    beam_size: int = 5
    vad: bool = True
    srt: bool = False
    language: str | None = None
    cleanup: bool = False
    project_root: Path = field(default_factory=lambda: Path("projects"))
    cookies_from_browser: str | None = None
    cookies_file: str | None = None


DEFAULT_CONFIG_TEMPLATE = """\
# transcribe — user configuration
# Uncomment any line to override the built-in default.
# Precedence (lowest to highest): builtin default < this file < TRANSCRIBE_* env vars < CLI flags.

# model = "turbo"                # turbo | large-v3 | medium | small | tiny
# device = "auto"                # auto | cuda | cpu
# compute_type = "float16"       # leave unset for a device-appropriate default
# beam_size = 5
# vad = true
# srt = false
# language = "en"                # leave unset for auto-detect
# cleanup = false
# project_root = "projects"           # relative to cwd; use an absolute path to fix it globally
# cookies_from_browser = "chrome"     # browser to pull cookies from: chrome, firefox, chromium, edge, safari
# cookies_file = ""                   # path to a Netscape-format cookies.txt file
"""


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on", "y")


_ENV_MAP: dict[str, tuple[str, Callable[[str], object]]] = {
    "TRANSCRIBE_MODEL": ("model", str),
    "TRANSCRIBE_DEVICE": ("device", str),
    "TRANSCRIBE_COMPUTE_TYPE": ("compute_type", str),
    "TRANSCRIBE_BEAM_SIZE": ("beam_size", int),
    "TRANSCRIBE_VAD": ("vad", _parse_bool),
    "TRANSCRIBE_SRT": ("srt", _parse_bool),
    "TRANSCRIBE_LANGUAGE": ("language", str),
    "TRANSCRIBE_CLEANUP": ("cleanup", _parse_bool),
    "TRANSCRIBE_PROJECT_ROOT": ("project_root", lambda s: Path(s).expanduser()),
    "TRANSCRIBE_COOKIES_FROM_BROWSER": ("cookies_from_browser", str),
    "TRANSCRIBE_COOKIES_FILE": ("cookies_file", str),
}


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "transcribe" / "config.toml"


def load_file(path: Path) -> dict:
    """Read TOML config. Returns {} if the file does not exist. Writes a template on first run."""
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
        except OSError:
            pass
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if "project_root" in raw:
        raw["project_root"] = Path(raw["project_root"]).expanduser()
    return raw


def load_env(env: dict[str, str] | None = None) -> dict:
    src = env if env is not None else os.environ
    out: dict = {}
    for env_name, (key, cast) in _ENV_MAP.items():
        if env_name in src:
            try:
                out[key] = cast(src[env_name])
            except (ValueError, TypeError):
                continue
    return out


def merge(file_cfg: dict, env_cfg: dict) -> dict:
    """Merge layered config into a flat dict using builtin defaults as the base."""
    defaults = asdict(Settings())
    return {**defaults, **file_cfg, **env_cfg}
