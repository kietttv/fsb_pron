"""Step 5: deterministic color mapping (FSP_Probosal.md §3.5)."""

from __future__ import annotations

from enum import Enum
from html import escape
from typing import Iterable, List, Sequence, Tuple

from fsb_pron.config import ThresholdConfig


class FeedbackTier(Enum):
    CORRECT = "correct"  # green
    ACCEPTABLE = "acceptable"  # yellow
    INCORRECT = "incorrect"  # red


TIER_COLOR = {
    FeedbackTier.CORRECT: "#2e7d32",
    FeedbackTier.ACCEPTABLE: "#f9a825",
    FeedbackTier.INCORRECT: "#c62828",
}


def gop_to_tier(gop: float, cfg: ThresholdConfig | None = None) -> FeedbackTier:
    cfg = cfg or ThresholdConfig()
    if gop >= cfg.tau_high:
        return FeedbackTier.CORRECT
    if gop >= cfg.tau_low:
        return FeedbackTier.ACCEPTABLE
    return FeedbackTier.INCORRECT


def tune_thresholds_from_dev_gops(
    gops: Sequence[float],
    *,
    target_green_frac: float = 0.35,
    target_yellow_frac: float = 0.40,
) -> ThresholdConfig:
    """
    Simple quantile-based starting point on L2-ARCTIC calibration split.
    Replace with proper optimization when expert labels exist.
    """
    arr = sorted(float(x) for x in gops)
    if not arr:
        return ThresholdConfig()
    n = len(arr)
    i_high = max(0, min(n - 1, int((1.0 - target_green_frac) * n)))
    i_low = max(0, min(n - 1, int((1.0 - target_green_frac - target_yellow_frac) * n)))
    return ThresholdConfig(tau_high=arr[i_high], tau_low=arr[i_low])


def render_learner_dashboard_html(
    phones: Sequence[str],
    gops: Sequence[float],
    cfg: ThresholdConfig | None = None,
    title: str = "Learner feedback",
) -> str:
    """Minimal HTML prototype for phoneme-colored transcript."""
    cfg = cfg or ThresholdConfig()
    parts: List[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        "<style>body{font-family:system-ui;margin:24px;background:#fafafa;}",
        ".row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:16px;}",
        ".ph{padding:6px 10px;border-radius:8px;color:#fff;font-weight:600;font-size:14px;}",
        ".leg span{margin-right:16px;font-size:13px;}</style></head><body>",
        f"<h2>{escape(title)}</h2>",
        "<div class='leg'><span style='color:#2e7d32'>■ Correct</span>",
        "<span style='color:#f9a825'>■ Acceptable</span>",
        "<span style='color:#c62828'>■ Incorrect</span></div><div class='row'>",
    ]
    for p, g in zip(phones, gops):
        tier = gop_to_tier(g, cfg)
        col = TIER_COLOR[tier]
        label = escape(f"{p}")
        tip = escape(f"GOP={g:.3f}")
        parts.append(
            f"<span class='ph' style='background:{col}' title='{tip}'>{label}</span>"
        )
    parts.append("</div></body></html>")
    return "".join(parts)
