import json
from collections.abc import Iterable
from pathlib import Path

from .utils import srt_timestamp


def write_txt(segments: Iterable, output_path: Path) -> None:
    parts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            parts.append(text)
    transcript = "\n\n".join(parts).strip()
    output_path.write_text((transcript + "\n") if transcript else "", encoding="utf-8")


def write_srt(segments: list, output_path: Path) -> None:
    lines: list[str] = []
    index = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        index += 1
        start = srt_timestamp(segment.start)
        end = srt_timestamp(segment.end)
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(segments: list, info: object, output_path: Path) -> None:
    """Write a simple structured JSON transcript (coexists with .txt/.srt).

    Includes top-level info from Whisper + per-segment data.
    If word_timestamps were enabled, includes "words" arrays on segments.
    """
    segs: list[dict] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        d: dict = {
            "start": segment.start,
            "end": segment.end,
            "text": text,
        }
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
