"""Paths and tunables aligned with `.agents/rules/context.md` and FSP_Probosal.md §3.5."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

# Kaggle defaults from context.md; override with env for local runs.
DATASET_PATHS: Dict[str, str] = {
    "l2_arctic": os.environ.get(
        "FSB_L2_ARCTIC",
        "/kaggle/input/datasets/adshyanmatheetharan/l2-arctic",
    ),
    "librispeech": os.environ.get(
        "FSB_LIBRISPEECH",
        "/kaggle/input/datasets/victorling/librispeech-clean/LibriSpeech/train-clean-100",
    ),
}

SAMPLING_RATE = 16_000


@dataclass(frozen=True)
class ThresholdConfig:
    """FSP_Probosal.md §3.5 example thresholds (tune on L2-ARCTIC calibration split)."""

    tau_high: float = -0.1  # Green if GOP >= tau_high
    tau_low: float = -0.5   # Red if GOP < tau_low; Yellow between


@dataclass(frozen=True)
class SplitConfig:
    """L2 buckets for finetune / calibration / PCC hold-out."""

    # Cumulative proportions on hash bucket [0, 1)
    frac_l2_finetune: float = 0.60
    frac_l2_calibration: float = 0.80  # up to this after finetune slice; rest = pcc
    # "speaker": same speaker → same split (speaker-disjoint eval). "utterance": hash speaker+utt
    # (needed when manifest has only 1–2 L2 speakers, else one bucket may stay empty).
    l2_stratify: str = "utterance"
