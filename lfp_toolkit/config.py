# lfp_toolkit/config.py

from __future__ import annotations

from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field

class CleaningConfig(BaseModel):
    """Configuration for LFP cleaning pipeline."""
    bandpass_l_freq: float | None = 1.0
    bandpass_h_freq: float | None = 100.0
    notch_freq: float | None = 60.0
    notch_max_db: float = 1.0
    notch_sigma_hz: float = 2.0
    
    ekg_thr: float | str = 50e-6
    ekg_auto_sensitivity: float = 2.0
    ekg_polarity: int = 1
    subtract_t_wave: bool = False
    qrs_highpass_hz: float = 4.0
    
    dead_segment_sec: float = 0.2
    dead_std_ratio: float = 0.05
    min_gap_merge_sec: float = 1.0
    min_dead_duration_sec: float = 0.3
    reconstruct_dead: bool = True
    recon_context_sec: float = 2.0
    
    spike_segment_sec: float = 0.5
    spike_mad_thresh: float = 8.0

class EpochingConfig(BaseModel):
    """Configuration for LFP epoching pipeline."""
    output_dir: Path
    lfp_file: Path
    file_format: str | None = None  # 'edf', 'egi', 'fif', 'brainvision' or None for auto
    events_tsv: Path
    alignment_column: str
    subject_id: str
    epoch_tmin: float = -1.5
    epoch_tmax: float = 1.0
    fade_duration_sec: float = 0.2
    target_sfreq: float | None = None

class TFRConfig(BaseModel):
    """Configuration for TFR computation and plotting."""
    tensor_file: Path
    output_base: Path
    subject_id: str
    condition: str
    epoch_tmin: float = -1.5
    epoch_tmax: float = 1.0
    freqs: list[float] = Field(default_factory=lambda: np.arange(4, 91, 2).tolist())
    n_cycles_divisor: float = 3.0
    baseline_window: tuple[float, float] = (-1.5, -1.0)
    plot_tmin: float = -1.0
    plot_tmax: float = 1.0
    plot_fmin: float = 4.0
    plot_fmax: float = 50.0
    vlim: tuple[float, float] = (-1.0, 1.0)
    anatomical_labels: dict[str, str] = Field(default_factory=dict)