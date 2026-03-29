"""PCC between system GOP and expert scores (context.md) on L2-ARCTIC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExpertPhoneScore:
    """One row in expert_labels.jsonl."""

    utterance_id: str
    phoneme_index: int
    expert_score: float
    speaker: Optional[str] = None


def load_expert_jsonl(path: str) -> List[ExpertPhoneScore]:
    rows: List[ExpertPhoneScore] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d: Dict[str, Any] = json.loads(line)
            rows.append(
                ExpertPhoneScore(
                    utterance_id=d["utterance_id"],
                    phoneme_index=int(d["phoneme_index"]),
                    expert_score=float(d["expert_score"]),
                    speaker=d.get("speaker"),
                )
            )
    return rows


def pearson_gop_expert(
    gop_values: np.ndarray,
    expert_values: np.ndarray,
) -> Tuple[float, float]:
    """Returns (pearson_r, p_value). Requires scipy."""
    from scipy.stats import pearsonr

    m = np.isfinite(gop_values) & np.isfinite(expert_values)
    if m.sum() < 2:
        return float("nan"), float("nan")
    r, p = pearsonr(gop_values[m], expert_values[m])
    return float(r), float(p)


def align_scores_by_key(
    gop_by_key: Dict[Tuple[str, int], float],
    experts: List[ExpertPhoneScore],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build parallel arrays for keys present in both."""
    gs: List[float] = []
    es: List[float] = []
    for ex in experts:
        k = (ex.utterance_id, ex.phoneme_index)
        if k in gop_by_key:
            gs.append(gop_by_key[k])
            es.append(ex.expert_score)
    return np.array(gs, dtype=np.float64), np.array(es, dtype=np.float64)


def expert_labels_schema_doc() -> str:
    return (
        "Expert labels file: JSONL, one object per line.\n"
        'Required keys: "utterance_id" (str), "phoneme_index" (int, 0-based in alignment), '
        '"expert_score" (float, e.g. goodness 1–5).\n'
        'Optional: "speaker".\n'
        "Join to system output via (utterance_id, phoneme_index) after same alignment order."
    )
