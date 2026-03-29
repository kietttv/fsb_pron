"""Build unified manifest: L2-ARCTIC (TextGrid phones + transcript) + LibriSpeech (G2P from text)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional

import librosa
from tqdm.auto import tqdm

from fsb_pron.config import DATASET_PATHS
from fsb_pron.phoneme_utils import phones_from_textgrid_label
from fsb_pron.textgrid import load_textgrid_intervals, pick_textgrid_for_utterance


@dataclass
class ManifestRecord:
    audio_path: str
    phonemes: str
    corpus: str
    speaker: str
    utterance_id: str
    transcript: str
    duration_sec: float
    split_role: str = "finetune"  # set by splits.assign_split_role

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _l2_phones_from_textgrid(tg_path: str) -> List[str]:
    phones: List[str] = []
    for it in load_textgrid_intervals(tg_path, tier_name="phones"):
        phones.extend(phones_from_textgrid_label(it.text))
    return phones


def _read_l2_transcript(speaker_dir: str, utt_id: str) -> str:
    tp = os.path.join(speaker_dir, "transcript", f"{utt_id}.txt")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    return ""


def ensure_nltk_for_g2p() -> None:
    """Download NLTK data used by g2p_en (pos_tag). Safe to call repeatedly."""
    try:
        import nltk
    except ImportError:
        return
    for pkg in (
        "averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger",
        "punkt",
        "punkt_tab",
    ):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def iter_l2_arctic_records(
    root: str,
    *,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    prefer_annotation: bool = False,
) -> Iterator[ManifestRecord]:
    if not os.path.isdir(root):
        return
    for speaker in sorted(os.listdir(root)):
        sp_dir = os.path.join(root, speaker)
        if not os.path.isdir(sp_dir):
            continue
        wav_dir = os.path.join(sp_dir, "wav")
        if not os.path.isdir(wav_dir):
            continue
        for fname in os.listdir(wav_dir):
            if not fname.lower().endswith(".wav"):
                continue
            utt_id = fname[:-4]
            wav_path = os.path.join(wav_dir, fname)
            tg = pick_textgrid_for_utterance(sp_dir, utt_id, prefer_annotation=prefer_annotation)
            if not tg:
                continue
            plist = _l2_phones_from_textgrid(tg)
            if not plist:
                continue
            try:
                dur = float(librosa.get_duration(path=wav_path))
            except Exception:
                continue
            if not (min_duration <= dur <= max_duration):
                continue
            yield ManifestRecord(
                audio_path=os.path.abspath(wav_path),
                phonemes=" ".join(plist),
                corpus="l2_arctic",
                speaker=speaker,
                utterance_id=utt_id,
                transcript=_read_l2_transcript(sp_dir, utt_id),
                duration_sec=round(dur, 3),
            )


def iter_librispeech_records(
    root: str,
    *,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    g2p=None,
) -> Iterator[ManifestRecord]:
    """
    LibriSpeech train-clean-100 layout: root/<reader>/<chapter>/*.flac + *.trans.txt
    Phoneme string from g2p_en.G2p if g2p is provided.
    """
    if not os.path.isdir(root) or g2p is None:
        return
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".trans.txt"):
                continue
            trans_path = os.path.join(dirpath, fn)
            with open(trans_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) < 2:
                        continue
                    utt_id, text = parts[0], parts[1]
                    # utt_id like 103-1240-0000 -> flac path
                    flac_path = os.path.join(dirpath, f"{utt_id}.flac")
                    if not os.path.isfile(flac_path):
                        continue
                    try:
                        dur = float(librosa.get_duration(path=flac_path))
                    except Exception:
                        continue
                    if not (min_duration <= dur <= max_duration):
                        continue
                    raw_phones = [p for p in g2p(text) if p]
                    from fsb_pron.phoneme_utils import clean_g2p_sequence

                    plist = clean_g2p_sequence(raw_phones)
                    if not plist:
                        continue
                    chap = os.path.basename(dirpath)
                    reader = os.path.basename(os.path.dirname(dirpath))
                    yield ManifestRecord(
                        audio_path=os.path.abspath(flac_path),
                        phonemes=" ".join(plist),
                        corpus="librispeech",
                        speaker=f"{reader}_{chap}",
                        utterance_id=utt_id,
                        transcript=text,
                        duration_sec=round(dur, 3),
                    )


def build_manifest(
    *,
    l2_root: Optional[str] = None,
    librispeech_root: Optional[str] = None,
    out_jsonl: str,
    g2p_engine=None,
    max_l2: Optional[int] = None,
    max_libri: Optional[int] = None,
    show_progress: bool = True,
    assign_splits: bool = True,
) -> List[ManifestRecord]:
    l2_root = l2_root or DATASET_PATHS["l2_arctic"]
    librispeech_root = librispeech_root or DATASET_PATHS["librispeech"]

    records: List[ManifestRecord] = []

    l2_it = list(iter_l2_arctic_records(l2_root))
    if max_l2 is not None:
        l2_it = l2_it[:max_l2]
    records.extend(l2_it)

    if g2p_engine is not None:
        ensure_nltk_for_g2p()
        lib_it = list(iter_librispeech_records(librispeech_root, g2p=g2p_engine))
        if max_libri is not None:
            lib_it = lib_it[:max_libri]
        records.extend(lib_it)

    if assign_splits:
        from fsb_pron.splits import apply_splits_to_records

        apply_splits_to_records(records)

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    it_write = tqdm(records, desc="Writing manifest") if show_progress else records
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in it_write:
            f.write(json.dumps(r.to_json_dict(), ensure_ascii=False) + "\n")
    return records
