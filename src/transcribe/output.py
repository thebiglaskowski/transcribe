from pathlib import Path
from typing import Iterable

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
