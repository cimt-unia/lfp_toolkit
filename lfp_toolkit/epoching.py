# lfp_toolkit/epoching.py

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import resample
from scipy.signal.windows import hann

from .config import EpochingConfig
from .cleaning import clean_lfp
from .io import load_raw

logger = logging.getLogger(__name__)

def load_validated_events(events_path: Path, alignment_col: str) -> pd.DataFrame:
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    df = pd.read_csv(events_path, sep="\t")
    if "outcome_label" in df.columns:
        nan_mask = df["outcome_label"].isna()
        n_excluded = int(nan_mask.sum())
        if n_excluded > 0:
            logger.warning("Excluding %d trials with NaN outcome_label", n_excluded)
        df = df[~nan_mask].reset_index(drop=True)
        
    if df.empty:
        raise RuntimeError("No valid trials remaining after NaN exclusion.")

    if alignment_col not in df.columns:
        raise ValueError(f"Alignment column '{alignment_col}' not found in events TSV.")

    valid = df.sort_values(alignment_col).reset_index(drop=True)
    logger.info("Loaded %d validated cue trials", len(valid))
    return valid

def extract_and_reject_epochs(
    data: NDArray[np.float64], dead_masks: NDArray[np.bool_], sfreq: float,
    event_onsets_sec: NDArray[np.float64], tmin: float, tmax: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    n_channels, n_times = data.shape
    epoch_samples = int(round((tmax - tmin) * sfreq))
    offset_samples = int(round(tmin * sfreq))

    onset_samples = np.round(event_onsets_sec * sfreq).astype(int)
    starts = onset_samples + offset_samples
    ends = starts + epoch_samples

    out_of_bounds = (starts < 0) | (ends > n_times)
    if out_of_bounds.any():
        bad_indices = np.where(out_of_bounds)[0]
        raise ValueError(f"{len(bad_indices)} epochs exceed signal boundaries.")

    keep_mask = np.ones(len(onset_samples), dtype=bool)
    for i, (start, end) in enumerate(zip(starts, ends)):
        if dead_masks[:, start:end].any():
            keep_mask[i] = False

    kept_indices = np.where(keep_mask)[0]
    n_rejected = len(onset_samples) - len(kept_indices)
    logger.info("Epoch rejection: %d/%d kept, %d rejected", len(kept_indices), len(onset_samples), n_rejected)

    if len(kept_indices) == 0:
        raise RuntimeError("All epochs rejected. Inspect dead_masks or relax parameters.")

    epochs = np.empty((n_channels, len(kept_indices), epoch_samples), dtype=np.float64)
    for new_idx, orig_idx in enumerate(kept_indices):
        epochs[:, new_idx, :] = data[:, starts[orig_idx]:ends[orig_idx]]

    return epochs, kept_indices

def apply_epoch_fades(epochs: NDArray[np.float64], sfreq: float, fade_duration_sec: float) -> NDArray[np.float64]:
    n_channels, n_epochs, n_times = epochs.shape
    fade_samples = int(round(fade_duration_sec * sfreq))

    if fade_samples >= n_times // 2:
        raise ValueError("Fade duration exceeds half epoch length.")

    full_window = hann(2 * fade_samples, sym=True)
    fade_in = full_window[:fade_samples]
    fade_out = full_window[fade_samples:]

    faded = epochs.copy()
    faded[:, :, :fade_samples] *= fade_in[np.newaxis, np.newaxis, :]
    faded[:, :, -fade_samples:] *= fade_out[np.newaxis, np.newaxis, :]
    return faded

def convert_lfp_to_tensor(config: EpochingConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    events_df = load_validated_events(config.events_tsv, config.alignment_column)

    logger.info("Loading LFP from %s...", config.lfp_file.name)
    raw = load_raw(config.lfp_file, file_format=config.file_format)
    
    # Clean without reconstruction to match original strict rejection logic
    result = clean_lfp(
        raw, subject_id=config.subject_id, reconstruct_dead=False, 
        return_raw=False, return_removed=False, detect_spikes=False
    )
    
    data = result["data"][0]
    dead_masks = result["dead_mask"][0]
    source_channel_names = result["roi_names"]
    sfreq = result["sfreq"]

    recording_duration = data.shape[1] / sfreq
    valid_window = (
        (events_df[config.alignment_column] + config.epoch_tmin >= 0)
        & (events_df[config.alignment_column] + config.epoch_tmax <= recording_duration)
    )
    events_filtered = events_df[valid_window].reset_index(drop=True)
    if events_filtered.empty:
        raise RuntimeError("No trials remain after boundary filtering.")

    onsets = events_filtered[config.alignment_column].to_numpy(dtype=np.float64)
    epochs, kept_indices = extract_and_reject_epochs(
        data, dead_masks, sfreq, onsets, config.epoch_tmin, config.epoch_tmax,
    )

    epochs = apply_epoch_fades(epochs, sfreq, config.fade_duration_sec)

    if config.target_sfreq is not None and config.target_sfreq != sfreq:
        target_timepoints = int(round((config.epoch_tmax - config.epoch_tmin) * config.target_sfreq))
        epochs = resample(epochs, target_timepoints, axis=2)
        sfreq = config.target_sfreq

    tensor = epochs[np.newaxis, :, :, :]
    output_path = config.output_dir / f"tensor_{config.subject_id}_lfp_epochs.npz"
    
    np.savez_compressed(
        output_path, data=tensor, subject_ids=np.array([config.subject_id]),
        roi_names=np.array(source_channel_names), sfreq=sfreq,
        n_epochs=tensor.shape[2], n_rois=len(source_channel_names),
        tmin=config.epoch_tmin, tmax=config.epoch_tmax,
        alignment_column=config.alignment_column, kept_trial_indices=kept_indices,
        total_valid_trials=len(events_filtered),
        rejected_trials=len(events_filtered) - tensor.shape[2],
        fade_duration_sec=config.fade_duration_sec,
    )
    logger.info("Saved LFP tensor: %s → %s", tensor.shape, output_path)
    return output_path