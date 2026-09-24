import json
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest

from transcribe import config as cfg
from transcribe.cli import _classify_inputs
from transcribe.output import write_json, write_srt, write_txt
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


def test_write_json_basic(tmp_path: Path):
    segments = [
        _Segment(text="hi", start=0.0, end=1.5),
        _Segment(text="", start=1.5, end=2.0),  # blank, skipped
        _Segment(text="there", start=2.0, end=3.25),
    ]
    info = type("Info", (), {"language": "en", "language_probability": 0.99, "duration": 3.25})()
    out = tmp_path / "x.json"
    write_json(segments, info, out)  # type: ignore[arg-type]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["language"] == "en"
    assert data["duration"] == 3.25
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "hi"
    assert "words" not in data["segments"][0]  # no words unless provided by real segments


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


# ---------- CLI parser flags (version, list-models) ----------


def test_build_parser_has_version_and_list_models():
    from transcribe.cli import build_parser

    # build_parser requires the full merged defaults dict (used in help strings etc.)
    defaults = {
        "model": "turbo",
        "device": "auto",
        "compute_type": None,
        "beam_size": 5,
        "vad": True,
        "srt": False,
        "word_timestamps": False,
        "diarize": False,
        "language": None,
        "cleanup": False,
        "project_root": "projects",
        "cookies_from_browser": None,
        "cookies_file": None,
    }
    parser = build_parser(defaults)
    # version action exists (dest='version' for the action)
    assert any(getattr(a, "dest", None) == "version" for a in parser._actions)
    # list-models flag
    assert any(getattr(a, "dest", None) == "list_models" for a in parser._actions)


def test_list_models_flag_in_namespace():
    from transcribe.cli import build_parser

    defaults = {
        "model": "turbo",
        "device": "auto",
        "compute_type": None,
        "beam_size": 5,
        "vad": True,
        "srt": False,
        "word_timestamps": False,
        "language": None,
        "diarize": False,
        "cleanup": False,
        "project_root": "projects",
        "cookies_from_browser": None,
        "cookies_file": None,
    }
    parser = build_parser(defaults)
    args = parser.parse_args(["--list-models"])
    assert args.list_models is True


# ---------- config: env parsing + merge precedence ----------


def test_load_env_parses_types():
    env = {
        "TRANSCRIBE_MODEL": "small",
        "TRANSCRIBE_BEAM_SIZE": "7",
        "TRANSCRIBE_VAD": "no",
        "TRANSCRIBE_CLEANUP": "yes",
        "TRANSCRIBE_PROJECT_ROOT": "~/foo",
        "TRANSCRIBE_WORD_TIMESTAMPS": "true",
        "UNRELATED": "ignore-me",
    }
    out = cfg.load_env(env)
    assert out["model"] == "small"
    assert out["beam_size"] == 7
    assert out["vad"] is False
    assert out["cleanup"] is True
    assert out["project_root"] == Path("~/foo").expanduser()
    assert out["word_timestamps"] is True
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
    assert merged["word_timestamps"] is False


def test_load_file_creates_template_when_missing(tmp_path: Path):
    p = tmp_path / "config.toml"
    assert not p.exists()
    result = cfg.load_file(p)
    assert result == {}
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "# transcribe — user configuration" in text
    assert "# word_timestamps = false" in text


def test_load_file_expands_project_root_path(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('project_root = "~/custom-root"\n', encoding="utf-8")
    result = cfg.load_file(p)
    assert result["project_root"] == Path("~/custom-root").expanduser()


# ---------- terminal output (progress.py) ----------


def test_fmt_duration():
    from transcribe.progress import fmt_duration

    assert fmt_duration(None) == "0:00"
    assert fmt_duration(75.9) == "1:15"
    assert fmt_duration(3725) == "1:02:05"


def test_display_strips_width_ambiguous_emoji_modifiers():
    from transcribe.progress import display

    assert display("Dance 🕺\ufe0f 👨\u200d👩") == "Dance 🕺 👨👩"
    assert display(None) == ""


def test_live_title_drops_symbol_emoji_and_collapses_spaces():
    from transcribe.progress import live_title

    title = "Dance in space! 🪩🕺\u00a0🛰\ufe0f Microgravity"
    assert live_title(title) == "Dance in space! Microgravity"


def _recording_console(monkeypatch):
    from rich.console import Console

    import transcribe.progress as progress

    rec = Console(file=StringIO(), width=100, record=True, color_system=None)
    monkeypatch.setattr(progress, "console", rec)
    return rec


def test_console_handler_prefixes_levels_and_passes_text_through(monkeypatch):
    from rich.text import Text

    from transcribe.progress import ConsoleHandler

    rec = _recording_console(monkeypatch)
    log = logging.getLogger("transcribe.test_handler")
    log.handlers[:] = [ConsoleHandler()]
    log.propagate = False
    log.setLevel(logging.INFO)
    log.warning("slow [LIVE] %s", "down")  # brackets must survive (no markup parsing)
    log.error("broke")
    log.info(Text("✅ styled"))
    assert rec.export_text() == "🟡 slow [LIVE] down\n❌ broke\n✅ styled\n"


def test_tracker_runs_without_a_terminal(monkeypatch):
    from transcribe.progress import tracker

    _recording_console(monkeypatch)  # not a TTY → live region disabled, calls still work
    with tracker("Chan", 2) as t:
        t.start("A 🕺\ufe0f")
        t.stage("📥", 40)
        t.stage("📝")
        t.stage("📝", 10)
        t.advance()
        t.advance()
    assert t._overall_label() == "📺 Chan  [2/2]"


def test_progress_hook_reports_percent_and_skips_unknown_total():
    from transcribe.downloader import _make_progress_hook

    seen = []
    hook = _make_progress_hook(seen.append)
    hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 200})
    hook({"status": "downloading", "downloaded_bytes": 10})  # no total yet
    hook({"status": "finished"})
    assert seen == [25.0]


def test_done_line_formats_summary():
    from transcribe.transcribe import _done_line

    line = _done_line("Talk", {"duration": 751, "speakers": 2, "no_speech": False, "elapsed": 38})
    assert line.plain == "✅ Talk  12:31 · 👥 2 · ⚡ 0:38"
    quiet = _done_line("Clip", {"duration": 9, "speakers": None, "no_speech": True, "elapsed": 1})
    assert quiet.plain == "🔇 Clip  no speech"


# ---------- download retry on 403 ----------


def _fake_ydl(monkeypatch, tmp_path, failures):
    """Patch yt_dlp.YoutubeDL: raise a 403 DownloadError `failures` times, then write stem.wav."""
    import yt_dlp

    calls = []

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            calls.append(url)
            if len(calls) <= failures:
                raise yt_dlp.utils.DownloadError("ERROR: HTTP Error 403: Forbidden")
            (tmp_path / "clip.wav").write_bytes(b"x")
            return {"id": "abc"}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return calls


def test_download_audio_retries_after_403(monkeypatch, tmp_path):
    from transcribe.downloader import download_audio

    calls = _fake_ydl(monkeypatch, tmp_path, failures=1)
    assert download_audio("https://x", tmp_path, "clip") == (tmp_path / "clip.wav", {"id": "abc"})
    assert len(calls) == 2


def test_download_audio_gives_up_with_hint(monkeypatch, tmp_path):
    from transcribe.downloader import _DOWNLOAD_ATTEMPTS, download_audio

    calls = _fake_ydl(monkeypatch, tmp_path, failures=99)
    with pytest.raises(RuntimeError, match="uv tool upgrade transcribe"):
        download_audio("https://x", tmp_path, "clip")
    assert len(calls) == _DOWNLOAD_ATTEMPTS


# ---------- speaker labels ----------


def test_assign_speakers_majority_overlap_and_first_appearance_naming():
    from transcribe.transcribe import assign_speakers

    turns = [(0.0, 4.0, "SPEAKER_01"), (4.0, 10.0, "SPEAKER_00")]
    segs = [
        _Segment("a", 0.0, 3.0),  # all SPEAKER_01
        _Segment("b", 3.0, 7.0),  # straddles: 1s of 01, 3s of 00 -> 00
        _Segment("c", 20.0, 21.0),  # no overlap (music, silence)
    ]
    # SPEAKER_01 talks first, so it becomes "Speaker 1" regardless of pyannote's numbering
    assert assign_speakers(segs, turns) == ["Speaker 1", "Speaker 2", None]


def test_split_by_speaker_splits_mid_segment_and_keeps_unlabeled_words():
    from types import SimpleNamespace as W

    from transcribe.transcribe import split_by_speaker

    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    words = [
        W(word=" So", start=0.0, end=0.5),
        W(word=" yeah.", start=0.5, end=1.5),
        W(word=" Right.", start=2.5, end=3.0),
        W(word=" Uh", start=9.0, end=9.5),  # no turn: stays with the current run
    ]
    seg = W(text=" So yeah. Right. Uh", start=0.0, end=9.5, words=words)
    out, speakers = split_by_speaker([seg], turns)
    assert [s.text for s in out] == [" So yeah.", " Right. Uh"]
    assert speakers == ["Speaker 1", "Speaker 2"]
    assert (out[1].start, out[1].end) == (2.5, 9.5)


def test_write_txt_merges_consecutive_speaker_turns(tmp_path):
    segs = [_Segment("Hi.", 0, 1), _Segment("How are you?", 1, 2), _Segment("Good.", 2, 3)]
    out = tmp_path / "t.txt"
    write_txt(segs, out, ["Speaker 1", "Speaker 1", "Speaker 2"])
    assert out.read_text() == "Speaker 1: Hi. How are you?\n\nSpeaker 2: Good.\n"


# ---------- channels / title naming ----------


def test_video_stem_title_and_id():
    from transcribe.utils import video_stem

    assert video_stem("Is spider web stronger than steel?", "wt4p") == (
        "Is spider web stronger than steel [wt4p]"
    )


def test_video_stem_tiktok_caption_drops_hashtags_and_extra_lines():
    from transcribe.utils import video_stem

    assert video_stem("Dance in space! 🪩 #nasa #fyp\nmore", "768") == "Dance in space! 🪩 [768]"


def test_video_stem_strips_unsafe_chars_and_falls_back_to_id():
    from transcribe.utils import video_stem

    assert video_stem('a/b\\c: 100% [LIVE] "q" <x>|y*?.', "id1") == "a b c 100 LIVE q x y [id1]"
    assert video_stem("#fyp #viral", "123") == "123"
    assert video_stem(None, "123") == "123"


def test_video_stem_caps_bytes_without_splitting_emoji():
    from transcribe.utils import video_stem

    # 120 bytes = 30 four-byte emoji; a byte cut mid-emoji would leave invalid UTF-8
    assert video_stem("🚀" * 100, "x") == "🚀" * 30 + " [x]"


def test_list_sources_groups_tabs_profiles_and_single_videos(monkeypatch):
    import yt_dlp

    from transcribe.downloader import list_sources

    infos = {
        "channel": {
            "title": "Chan",
            "entries": [
                {"title": "Chan - Videos", "entries": [{"id": "v1", "title": "One", "url": "u1"}]},
                {"title": "Chan - Live", "entries": []},  # empty tab is dropped
                {"title": "Chan - Shorts", "entries": [{"id": "s1", "title": "S", "url": "u2"}]},
            ],
        },
        "profile": {"title": "tt", "entries": [{"id": "t1", "title": "T", "url": "u3"}, None]},
        "single": {"id": "x", "title": "X", "webpage_url": "https://w/x", "url": "media"},
    }

    class FakeYDL:
        def __init__(self, opts):
            assert opts["extract_flat"] == "in_playlist"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            assert download is False
            return infos[url]

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    tabs = list_sources("channel")
    assert [(t, [v["id"] for v in vids]) for t, vids in tabs] == [
        ("Chan - Videos", ["v1"]),
        ("Chan - Shorts", ["s1"]),
    ]
    assert list_sources("profile") == [("tt", [{"id": "t1", "title": "T", "url": "u3"}])]
    assert list_sources("single") == [(None, [{"id": "x", "title": "X", "url": "https://w/x"}])]


def test_prompt_pick_sources_parses_choices(monkeypatch, capsys):
    from transcribe.prompts import prompt_pick_sources

    groups = [("Videos", [1, 2]), ("Live", [3]), ("Shorts", [4])]
    answers = iter(["4", "x", "3, 1"])  # out of range, junk, then a valid pick
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert prompt_pick_sources("url", groups) == [groups[0], groups[2]]
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt_pick_sources("url", groups) == groups


def test_source_header_and_txt_header(tmp_path):
    from transcribe.output import source_header

    header = source_header(
        {"title": "T\nx", "channel": "C", "upload_date": "20260831", "webpage_url": "https://u"}
    )
    assert header == "Title: T x\nChannel: C\nUploaded: 2026-08-31\nURL: https://u"
    out = tmp_path / "t.txt"
    write_txt([_Segment("Hi.", 0, 1)], out, ["Speaker 1"], header)
    assert out.read_text() == f"{header}\n---\n\nSpeaker 1: Hi.\n"


def test_short_error_trims_ytdlp_preamble_and_adds_hint():
    from transcribe.transcribe import short_error

    age = RuntimeError(
        "\x1b[0;31mERROR:\x1b[0m [youtube] TnjbV97pV_c: Sign in to confirm your age."
        " Use --cookies-from-browser or --cookies for the authentication. See https://github.com/yt-dlp/...\nmore"
    )
    assert short_error(age) == (
        "Sign in to confirm your age · 🔞 age-restricted: needs YouTube cookies (see README)"
    )
    assert short_error(ValueError("boom")) == "boom"
    assert short_error(ValueError()) == "ValueError"


def test_tracker_unmeasurable_step_clears_total(monkeypatch):
    from transcribe.progress import tracker

    _recording_console(monkeypatch)
    with tracker("Chan", 1) as t:
        t.start("A")
        t.stage("📝", 50)
        t.stage("👥")  # speaker labels report no percentage
        item = next(task for task in t._progress.tasks if task.id == t._item)
        assert item.total is None and item.description == "👥 A"


def _run_failing_project(monkeypatch, tmp_path, errors):
    """Run _process_project where each download raises the next message from `errors`."""
    import argparse

    import transcribe.transcribe as tr

    messages = iter(errors)

    def fake_download(*a, **kw):
        raise RuntimeError(f"ERROR: [youtube] abc: {next(messages)}")

    monkeypatch.setattr(tr, "download_audio", fake_download)
    _recording_console(monkeypatch)
    videos = [{"title": f"v{i}", "url": "u", "stem": f"v{i}"} for i in range(len(errors))]
    args = argparse.Namespace(srt=False, json=False, cleanup=False)
    return tr._process_project(tmp_path, videos, args, None, None, {})


def test_project_stops_after_repeated_run_wide_failures(monkeypatch, tmp_path):
    errors = ["The page needs to be reloaded."] * 8
    assert _run_failing_project(monkeypatch, tmp_path, errors) == (0, 5, True)


def test_project_does_not_stop_for_per_video_failures(monkeypatch, tmp_path):
    # age-restricted videos neither trip the stop nor break a run-wide streak
    errors = ["Sorry, this content is age-restricted."] * 6
    errors += ["The page needs to be reloaded.", "Sorry, this content is age-restricted."]
    assert _run_failing_project(monkeypatch, tmp_path, errors) == (0, 8, False)
