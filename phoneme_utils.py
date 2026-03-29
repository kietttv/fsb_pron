"""Normalize phone strings for CTC vocab (ARPAbet-style)."""

from __future__ import annotations

import re
from typing import Iterable, List

SILENCE_PHONES = {
    "h#",
    "pau",
    "epi",
    "q",
    "bcl",
    "dcl",
    "gcl",
    "pcl",
    "tcl",
    "kcl",
    "sil",
    "sp",
    "spn",
    "SIL",
    "<eps>",
    "!",
    "<unk>",
    "''",
    "",
}
PUNCT = {".", ",", "!", "?", ";", ":", "-", "'", ""}


def phones_from_textgrid_label(raw: str) -> List[str]:
    """Turn one TextGrid phone label into zero or one ARPAbet tokens (L2 error tags)."""
    m = raw.strip()
    if not m or m in SILENCE_PHONES:
        return []
    if "," in m:
        parts = [p.strip() for p in m.split(",")]
        if parts and parts[0].lower() == "sil":
            return []
        m = parts[0]
    m = re.sub(r"[\d*]", "", m).strip()
    if not m:
        return []
    return [m.upper()]


def clean_g2p_sequence(tokens: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in tokens:
        p = str(p).strip()
        p = re.sub(r"\d", "", p)
        pl = p.lower()
        if not pl or pl in SILENCE_PHONES or pl in PUNCT:
            continue
        out.append(pl.upper())
    return out
