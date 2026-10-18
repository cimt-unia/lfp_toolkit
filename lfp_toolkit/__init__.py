# lfp_toolkit/__init__.py

from .config import CleaningConfig, EpochingConfig, TFRConfig
from .io import load_raw
from .cleaning import clean_lfp, LFPCleanResult
from .viz import viz_data
from .epoching import convert_lfp_to_tensor
from .tfr import generate_tfr_plots, compute_roi_tfr

__all__ = [
    "CleaningConfig",
    "EpochingConfig",
    "TFRConfig",
    "load_raw",
    "clean_lfp",
    "LFPCleanResult",
    "viz_data",
    "convert_lfp_to_tensor",
    "generate_tfr_plots",
    "compute_roi_tfr",
]