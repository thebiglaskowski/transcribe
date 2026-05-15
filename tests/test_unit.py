from dataclasses import dataclass
from pathlib import Path

import pytest

from transcribe import config as cfg
from transcribe.cli import _classify_inputs
from transcribe.output import write_srt, write_txt
from transcribe.utils import is_url, sanitize_project_name, srt_timestamp


@dataclass
class _Segment:
    text: str
    start: float = 0.0
    end: float = 0.0


# ---------- srt_timestamp ----------


def test_srt_timestamp_zero():
    assert srt_timestamp(0) == "00:00:00,000"


def test_srt_timestamp_hours_minutes_seconds_ms():
    # 1h 1m 1.5s = 3661.5 -> 01:01:01,500
    assert srt_timestamp(3661.5) == "01:01:01,500"


def test_srt_timestamp_sub_second_rounds():
    # round() uses banker's rounding, so we pick values that aren't on a .5 boundary.
    assert srt_timestamp(0.001) == "00:00:00,001"
    assert srt_timestamp(0.0006) == "00:00:00,001"
    assert srt_timestamp(0.0004) == "00:00:00,000"


# ---------- write_txt / write_srt ----------


def test_write_txt_blanks_separated_by_blank_line(tmp_path: Path):
    segments = [_Segment(text="  hello  "), _Segment(text="world"), _Segment(text="   ")]
    out = tmp_path / "x.txt"
    write_txt(segments, out)
    assert out.read_text(encoding="utf-8") == "hello\n\nworld\n"


def test_write_txt_empty_writes_empty_file(tmp_path: Path):
    out = tmp_path / "x.txt"
    write_txt([_Segment(text="   ")], out)
    assert out.read_text(encoding="utf-8") == ""


def test_write_srt_numbers_and_timestamps(tmp_path: Path):
    segments = [
        _Segment(text="hi", start=0.0, end=1.5),
        _Segment(text="", start=1.5, end=2.0),  # blank, skipped
        _Segment(text="there", start=2.0, end=3.25),
    ]
    out = tmp_path / "x.srt"
    write_srt(segments, out)
    content = out.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,500\nhi" in content
    assert "2\n00:00:02,000 --> 00:00:03,250\nthere" in content
    # The blank segment must not appear and must not bump the index.
    assert "3\n" not in content


# ---------- is_url ----------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://example.com", True),
        ("https://youtu.be/abc", True),
        ("youtu.be/abc", False),
        ("/mnt/c/foo.wav", False),
        ("ftp://example.com/file.mp3", False),
        ("", False),
    ],
)
def test_is_url(value, expected):
    assert is_url(value) is expected


# ---------- sanitize_project_name ----------


def test_sanitize_keeps_word_chars_and_dashes():
    assert sanitize_project_name("my research-2026") == "my-research-2026"


def test_sanitize_collapses_consecutive_specials():
    assert sanitize_project_name("a / b \\ c") == "a---b---c"


def test_sanitize_strips_leading_trailing_dashes():
    assert sanitize_project_name("///hello///") == "hello"


def test_sanitize_unicode_word_chars_kept():
    # \w in Python's re matches unicode letters by default.
    assert sanitize_project_name("café 2026") == "café-2026"


def test_sanitize_empty_returns_none():
    assert sanitize_project_name("") is None
    assert sanitize_project_name("   ") is None


def test_sanitize_all_invalid_returns_none():
    assert sanitize_project_name("///") is None
    assert sanitize_project_name("!!!") is None


# ---------- _classify_inputs ----------


def test_classify_urls_only():
    urls, paths, mixed = _classify_inputs(["https://a.com", "http://b.com"])
    assert urls == ["https://a.com", "http://b.com"]
    assert paths == []
    assert mixed is False


def test_classify_paths_only():
    urls, paths, mixed = _classify_inputs(["a.wav", "b.m4a"])
    assert urls == []
    assert paths == ["a.wav", "b.m4a"]
    assert mixed is False


def test_classify_mixed():
    urls, paths, mixed = _classify_inputs(["https://a.com", "b.wav"])
    assert urls == ["https://a.com"]
    assert paths == ["b.wav"]
    assert mixed is True


# ---------- config: env parsing + merge precedence ----------


def test_load_env_parses_types():
    env = {
        "TRANSCRIBE_MODEL": "small",
        "TRANSCRIBE_BEAM_SIZE": "7",
        "TRANSCRIBE_VAD": "no",
        "TRANSCRIBE_CLEANUP": "yes",
        "TRANSCRIBE_PROJECT_ROOT": "~/foo",
        "UNRELATED": "ignore-me",
    }
    out = cfg.load_env(env)
    assert out["model"] == "small"
    assert out["beam_size"] == 7
    assert out["vad"] is False
    assert out["cleanup"] is True
    assert out["project_root"] == Path("~/foo").expanduser()
    assert "UNRELATED" not in out


def test_load_env_skips_bad_casts():
    out = cfg.load_env({"TRANSCRIBE_BEAM_SIZE": "not-an-int"})
    assert "beam_size" not in out


def test_merge_env_overrides_file_overrides_defaults():
    file_cfg = {"model": "medium", "srt": True, "beam_size": 3}
    env_cfg = {"model": "small", "beam_size": 9}
    merged = cfg.merge(file_cfg, env_cfg)

    # env wins over file
    assert merged["model"] == "small"
    assert merged["beam_size"] == 9

    # file wins over default for keys env didn't supply
    assert merged["srt"] is True

    # default wins for keys neither layer supplied
    assert merged["vad"] is True


def test_load_file_creates_template_when_missing(tmp_path: Path):
    p = tmp_path / "config.toml"
    assert not p.exists()
    result = cfg.load_file(p)
    assert result == {}
    assert p.exists()
    assert "# transcribe — user configuration" in p.read_text(encoding="utf-8")


def test_load_file_expands_project_root_path(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('project_root = "~/custom-root"\n', encoding="utf-8")
    result = cfg.load_file(p)
    assert result["project_root"] == Path("~/custom-root").expanduser()
