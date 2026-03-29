"""
Step 2: alignment artifacts → frame indices for CTC / SSL-GOP.

MFA produces TextGrid intervals; use `fsb_pron.textgrid.load_textgrid_intervals`.
Map time (seconds) to model frame indices given `n_input_samples` and `n_output_frames`
from one forward pass (exact count avoids assumptions about CNN strides).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PhoneSegment:
    start_sec: float
    end_sec: float
    phone: str


def textgrid_to_segments(tg_path: str) -> List[PhoneSegment]:
    from fsb_pron.phoneme_utils import phones_from_textgrid_label
    from fsb_pron.textgrid import load_textgrid_intervals

    segs: List[PhoneSegment] = []
    for it in load_textgrid_intervals(tg_path, tier_name="phones"):
        toks = phones_from_textgrid_label(it.text)
        if not toks:
            continue
        phone = toks[0]
        segs.append(PhoneSegment(start_sec=it.start, end_sec=it.end, phone=phone))
    return segs


def time_span_to_frame_range(
    t0: float,
    t1: float,
    audio_duration_sec: float,
    n_frames: int,
) -> Tuple[int, int]:
    """Linear map [t0, t1] to inclusive-exclusive frame index range."""
    if audio_duration_sec <= 0 or n_frames <= 0:
        return 0, 0
    t0 = max(0.0, min(t0, audio_duration_sec))
    t1 = max(t0, min(t1, audio_duration_sec))
    i0 = int(np.floor(t0 / audio_duration_sec * n_frames))
    i1 = int(np.ceil(t1 / audio_duration_sec * n_frames))
    i0 = max(0, min(i0, n_frames))
    i1 = max(i0, min(i1, n_frames))
    return i0, i1


def segments_to_frame_spans(
    segments: List[PhoneSegment],
    audio_duration_sec: float,
    n_frames: int,
) -> List[Tuple[int, int, str]]:
    """List of (frame_start, frame_end, phone) with end exclusive."""
    out: List[Tuple[int, int, str]] = []
    for s in segments:
        a, b = time_span_to_frame_range(s.start_sec, s.end_sec, audio_duration_sec, n_frames)
        if b > a:
            out.append((a, b, s.phone))
    return out


def run_mfa_cli_doc() -> str:
    """Offline MFA (not invoked from Python). Typical workflow."""
    return (
        "1) conda install -c conda-forge montreal-forced-aligner; "
        "2) mfa model download acoustic english_us_arpa; "
        "3) mfa model download dictionary english_us_arpa; "
        "4) mfa align /path/to/wavs /path/to/corpus_dict.txt english_us_arpa english_us_arpa /path/to/out; "
        "5) consume TextGrids via fsb_pron.textgrid.load_textgrid_intervals."
    )
