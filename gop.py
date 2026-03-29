"""
Step 4: phoneme posteriorgram + GOP (FSP_Probosal.md §3.3–3.4).

GOP(p) = log P(p|O_t) - max_q log P(q|O_t) per frame; segment score = mean over frames in alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def posteriorgram_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """(T, V) logits -> (T, V) probabilities."""
    return F.softmax(logits, dim=-1)


def log_posteriorgram_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


@dataclass
class PhonemeGOPResult:
    phone: str
    gop: float
    frame_start: int
    frame_end: int


def gop_scores_for_segments(
    logits: torch.Tensor,
    frame_spans: Sequence[Tuple[int, int, str]],
    token_id_for_phone: Dict[str, int],
    unk_id: Optional[int] = None,
) -> List[PhonemeGOPResult]:
    """
    logits: (T, V) single utterance
    frame_spans: (start, end, phone) with end exclusive
    """
    lp = log_posteriorgram_from_logits(logits)  # (T, V)
    T, V = lp.shape
    max_lp = lp.max(dim=-1).values  # (T,)
    results: List[PhonemeGOPResult] = []
    for fs, fe, phone in frame_spans:
        fs = max(0, min(int(fs), T))
        fe = max(fs, min(int(fe), T))
        if fe <= fs:
            continue
        tid = token_id_for_phone.get(phone)
        if tid is None:
            tid = token_id_for_phone.get(phone.upper()) or token_id_for_phone.get(phone.lower())
        if tid is None and unk_id is not None:
            tid = unk_id
        if tid is None or tid >= V:
            continue
        chunk = lp[fs:fe]
        chunk_max = max_lp[fs:fe]
        per_frame = chunk[:, tid] - chunk_max
        gop = float(per_frame.mean().item())
        results.append(PhonemeGOPResult(phone=phone, gop=gop, frame_start=fs, frame_end=fe))
    return results


def numpy_logits_from_torch(logits: torch.Tensor) -> np.ndarray:
    return logits.detach().float().cpu().numpy()
