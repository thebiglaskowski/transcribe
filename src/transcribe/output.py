import json
from pathlib import Path

from .utils import srt_timestamp


def source_header(info: dict) -> str:
    """Title / channel / upload date / URL lines for the top of a downloaded video's transcript."""
    date = info.get("upload_date")  # YYYYMMDD
    fields = [
        ("Title", " ".join((info.get("title") or "").split())),
        ("Channel", info.get("channel") or info.get("uploader")),
        ("Uploaded", f"{date[:4]}-{date[4:6]}-{date[6:]}" if date else None),
        ("URL", info.get("webpage_url")),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields if value)


def write_txt(
    segments: list, output_path: Path, speakers: list | None = None, header: str | None = None
) -> None:
    """One paragraph per segment; with speakers, one "Speaker N: ..." paragraph per turn.

    An optional header (see source_header) goes above a "---" line.
    """
    parts: list[str] = [f"{header}\n---"] if header else []
    current = None
    for segment, speaker in zip(segments, speakers or [None] * len(segments), strict=True):
        text = segment.text.strip()
        if not text:
            continue
        if speakers is None:
            parts.append(text)
        elif speaker == current and parts and current is not None:
            parts[-1] += " " + text
        else:
            parts.append(f"{speaker}: {text}" if speaker else text)
            current = speaker
    transcript = "\n\n".join(parts).strip()
    output_path.write_text((transcript + "\n") if transcript else "", encoding="utf-8")


def write_srt(segments: list, output_path: Path, speakers: list | None = None) -> None:
    lines: list[str] = []
    index = 0
    for segment, speaker in zip(segments, speakers or [None] * len(segments), strict=True):
        text = segment.text.strip()
        if not text:
            continue
        if speaker:
            text = f"[{speaker}] {text}"
        index += 1
        start = srt_timestamp(segment.start)
        end = srt_timestamp(segment.end)
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    segments: list, info: object, output_path: Path, speakers: list | None = None
) -> None:
    """Write a simple structured JSON transcript (coexists with .txt/.srt).

    Includes top-level info from Whisper + per-segment data.
    If word_timestamps were enabled, includes "words" arrays on segments.
    """
    segs: list[dict] = []
    for segment, speaker in zip(segments, speakers or [None] * len(segments), strict=True):
        text = segment.text.strip()
        if not text:
            continue
        d: dict = {
            "start": segment.start,
            "end": segment.end,
            "text": text,
        }
        if speaker:
            d["speaker"] = speaker
        if hasattr(segment, "words") and segment.words:
            d["words"] = [
                {
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "probability": getattr(w, "probability", None),
                }
                for w in segment.words
            ]
        segs.append(d)

    data = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": segs,
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
