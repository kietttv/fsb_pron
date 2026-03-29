"""Parse TextGrid intervals (MFA or L2-ARCTIC textgrid / annotation tiers)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Interval:
    start: float
    end: float
    text: str


def _parse_textgrid_content(content: str, tier_name: str = "phones") -> List[Interval]:
    """Extract intervals from a Praat TextGrid string for a named interval tier."""
    low = content.lower()
    tn = tier_name.lower()
    idx = low.find(f'name = "{tn}"')
    if idx < 0:
        idx = content.find(f'name = "{tier_name}"')
    if idx < 0:
        return []
    block = content[idx:]
    # Stop at next tier header (next `name = "` after first line) if long format
    next_tier = re.search(r'\n\s*name\s*=\s*"', block[20:])
    if next_tier:
        block = block[: 20 + next_tier.start()]

    intervals: List[Interval] = []
    for im in re.finditer(
        r"xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*\"([^\"]*)\"",
        block,
        re.MULTILINE,
    ):
        start, end, text = float(im.group(1)), float(im.group(2)), im.group(3).strip()
        intervals.append(Interval(start=start, end=end, text=text))
    return intervals


def load_textgrid_intervals(path: str, tier_name: str = "phones") -> List[Interval]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return _parse_textgrid_content(content, tier_name=tier_name)


def pick_textgrid_for_utterance(
    speaker_dir: str,
    utt_id: str,
    prefer_annotation: bool = False,
) -> Optional[str]:
    """Return path to TextGrid if present (L2-ARCTIC layout: textgrid/ or annotation/)."""
    import os

    order = (
        ("annotation", "textgrid") if prefer_annotation else ("textgrid", "annotation")
    )
    names = (
        f"{utt_id}.TextGrid",
        f"{utt_id}.textgrid",
        f"{utt_id}.TEXTGRID",
    )
    for sub in order:
        tier_dir = os.path.join(speaker_dir, sub)
        if not os.path.isdir(tier_dir):
            continue
        for fn in names:
            p = os.path.join(tier_dir, fn)
            if os.path.isfile(p):
                return p
    return None
