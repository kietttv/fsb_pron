"""
Train / eval policy (plan: avoid leakage).

- LibriSpeech: all `finetune` (native reference for Phoneme Head).
- L2-ARCTIC: per-speaker hash buckets → `l2_finetune` | `l2_calibration` | `l2_pcc_eval`.
  Use calibration only to tune τ_high, τ_low; report PCC on `l2_pcc_eval` only.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from fsb_pron.config import SplitConfig
from fsb_pron.ingest import ManifestRecord

SplitRole = Literal["finetune", "l2_finetune", "l2_calibration", "l2_pcc_eval"]


def _speaker_bucket(speaker_id: str) -> float:
    h = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 1_000_000) / 1_000_000.0


def assign_split_role(
    record: ManifestRecord,
    cfg: SplitConfig | None = None,
) -> SplitRole:
    cfg = cfg or SplitConfig()
    if record.corpus == "librispeech":
        return "finetune"
    key = (
        record.speaker
        if cfg.l2_stratify == "speaker"
        else f"{record.speaker}:{record.utterance_id}"
    )
    r = _speaker_bucket(key)
    if r < cfg.frac_l2_finetune:
        return "l2_finetune"
    if r < cfg.frac_l2_calibration:
        return "l2_calibration"
    return "l2_pcc_eval"


def apply_splits_to_records(records: list[ManifestRecord], cfg: SplitConfig | None = None) -> None:
    for rec in records:
        role = assign_split_role(rec, cfg)
        # store on record.split_role for manifest consumers
        rec.split_role = role


def manifest_dict_split_role(row: dict) -> SplitRole:
    """Assign split from a JSONL row without full ManifestRecord."""
    from fsb_pron.ingest import ManifestRecord

    rec = ManifestRecord(
        audio_path=row["audio_path"],
        phonemes=row["phonemes"],
        corpus=row["corpus"],
        speaker=row["speaker"],
        utterance_id=row["utterance_id"],
        transcript=row.get("transcript", ""),
        duration_sec=float(row.get("duration_sec", 0.0)),
        split_role=row.get("split_role", "finetune"),
    )
    return assign_split_role(rec)
