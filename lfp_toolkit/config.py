# lfp_toolkit/config.py

from __future__ import annotations

from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator, ConfigDict

class CleaningConfig(BaseModel):
    """Configuration for LFP cleaning pipeline."""
    model_config = ConfigDict(extra="forbid")

    picks: str | list[str] = "eeg"
    bandpass_l_freq: float | None = 1.0
    bandpass_h_freq: float | None = 100.0
    notch_freq: float | None = 60.0
    notch_max_db: float = 1.0
    notch_sigma_hz: float = 2.0
    
    ekg_thr: float | Literal["auto"] = 50e-6
    ekg_auto_sensitivity: float = 2.0
    ekg_polarity: Literal[-1, 1] = 1
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

    @model_validator(mode="after")
    def check_freqs(self) -> "CleaningConfig":
        if self.bandpass_l_freq is not None and self.bandpass_h_freq is not None:
            if self.bandpass_l_freq >= self.bandpass_h_freq:
                raise ValueError("bandpass_l_freq must be < bandpass_h_freq")
        if self.notch_sigma_hz <= 0:
            raise ValueError("notch_sigma_hz must be > 0")
        return self

class EpochingConfig(BaseModel):
    """Configuration for LFP epoching pipeline."""
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    lfp_file: Path
    file_format: str | None = None
    events_tsv: Path
    alignment_column: str
    subject_id: str
    epoch_tmin: float = -1.5
    epoch_tmax: float = 1.0
    fade_duration_sec: float = 0.2
    target_sfreq: float | None = None
    padding_sec: float = 0.5  # Extra padding to protect TFR from wavelet edge effects
    cleaning_config: CleaningConfig | None = None

    @model_validator(mode="after")
    def check_times(self) -> "EpochingConfig":
        if self.epoch_tmin >= self.epoch_tmax:
            raise ValueError("epoch_tmin must be < epoch_tmax")
        return self

class TFRConfig(BaseModel):
    """Configuration for TFR computation and plotting."""
    model_config = ConfigDict(extra="forbid")

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
    show_plots: bool = False  # Prevents blocking in headless/batch environments
    save_plots: bool = True
