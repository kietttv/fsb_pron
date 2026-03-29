"""Compose preprocess → alignment → Wav2Vec2 logits → GOP → HTML feedback."""

from __future__ import annotations

from typing import List, Optional

from fsb_pron.alignment import segments_to_frame_spans, textgrid_to_segments
from fsb_pron.config import ThresholdConfig
from fsb_pron.gop import PhonemeGOPResult, gop_scores_for_segments
from fsb_pron.modeling import PhonemeCTCInference, build_token_id_map
from fsb_pron.ui_feedback import render_learner_dashboard_html


def e2e_phoneme_feedback_html(
    audio_path: str,
    textgrid_path: str,
    model_id_or_path: str,
    *,
    threshold_cfg: Optional[ThresholdConfig] = None,
    apply_input_zscore: bool = False,
) -> str:
    """
    Run inference + TextGrid alignment + GOP + color HTML in one call.
    Requires `transformers` and a Wav2Vec2 CTC checkpoint whose tokenizer matches phone symbols.
    """
    import librosa

    enc = PhonemeCTCInference(
        model_id_or_path,
        apply_input_zscore=apply_input_zscore,
    )
    logits = enc.logits_from_audio_path(audio_path)
    dur = float(librosa.get_duration(path=audio_path))
    n_frames = int(logits.shape[0])
    segs = textgrid_to_segments(textgrid_path)
    spans = segments_to_frame_spans(segs, dur, n_frames)
    tok2id = build_token_id_map(enc.processor)
    unk = enc.processor.tokenizer.unk_token_id
    results: List[PhonemeGOPResult] = gop_scores_for_segments(
        logits,
        spans,
        tok2id,
        unk_id=unk,
    )
    phones = [r.phone for r in results]
    gops = [r.gop for r in results]
    return render_learner_dashboard_html(phones, gops, threshold_cfg)
