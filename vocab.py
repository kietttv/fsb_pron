"""CTC vocab from manifest phoneme strings."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Set


def collect_phonemes_from_manifest_lines(jsonl_path: str) -> Set[str]:
    phones: Set[str] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            import json as _json

            row = _json.loads(line)
            seq = row.get("phonemes") or ""
            for p in seq.split():
                if p:
                    phones.add(p)
    return phones


def build_ctc_vocab(phones: Iterable[str]) -> Dict[str, int]:
    """Wav2Vec2 CTC: phoneme tokens + word delimiter + unk + pad."""
    vocab: Dict[str, int] = {}
    for i, p in enumerate(sorted(set(phones))):
        vocab[p] = i
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab[""] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
