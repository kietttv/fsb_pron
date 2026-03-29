"""
FSB pronunciation assessment pipeline (L2-ARCTIC + LibriSpeech, SSL-GOP).
"""

__version__ = "0.1.0"

from fsb_pron.config import DATASET_PATHS, SplitConfig, ThresholdConfig

__all__ = ["DATASET_PATHS", "SplitConfig", "ThresholdConfig", "__version__"]
