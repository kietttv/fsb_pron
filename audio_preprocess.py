"""Step 1: mono 16 kHz + zero-mean / unit-variance (context.md)."""

from __future__ import annotations

import numpy as np

from fsb_pron.config import SAMPLING_RATE


def waveform_zscore(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize 1D float waveform to zero mean and unit variance."""
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    m = float(x.mean())
    s = float(x.std())
    if s < eps:
        return (x - m).astype(np.float32)
    return ((x - m) / (s + eps)).astype(np.float32)


def load_waveform_mono_16k(
    path: str,
    *,
    target_sr: int = SAMPLING_RATE,
    apply_zscore: bool = True,
) -> np.ndarray:
    """Load audio file as mono float32 @ target_sr, optionally z-score."""
    import librosa

    y, _ = librosa.load(path, sr=target_sr, mono=True)
    y = y.astype(np.float32)
    if apply_zscore:
        y = waveform_zscore(y)
    return y
