"""Wav2Vec2 CTC phoneme model inference → logits for GOP (optional dependency: transformers)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class PhonemeCTCInference:
    def __init__(
        self,
        model_id_or_path: str,
        *,
        device: Optional[torch.device] = None,
        apply_input_zscore: bool = False,
    ):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path).to(self.device).eval()
        self.apply_input_zscore = apply_input_zscore

    @torch.inference_mode()
    def logits_from_waveform(
        self,
        waveform: np.ndarray,
        sampling_rate: int = 16_000,
    ) -> torch.Tensor:
        """Return (T, V) logits for a single utterance."""
        from fsb_pron.audio_preprocess import waveform_zscore

        w = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if self.apply_input_zscore:
            w = waveform_zscore(w)
        inputs = self.processor(w, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(self.device)
        mask = getattr(inputs, "attention_mask", None)
        if mask is not None:
            out = self.model(iv, attention_mask=mask.to(self.device)).logits
        else:
            out = self.model(iv).logits
        return out[0]

    def logits_from_audio_path(self, path: str, sampling_rate: int = 16_000) -> torch.Tensor:
        import librosa

        w, _ = librosa.load(path, sr=sampling_rate, mono=True)
        return self.logits_from_waveform(w.astype(np.float32), sampling_rate=sampling_rate)


def build_token_id_map(processor) -> dict:
    """Map phoneme string -> CTC label id from HF tokenizer vocab."""
    vocab = processor.tokenizer.get_vocab()
    return {k: v for k, v in vocab.items()}
