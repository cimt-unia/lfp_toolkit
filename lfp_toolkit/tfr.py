# lfp_toolkit/tfr.py

from __future__ import annotations
import logging
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from .config import TFRConfig

logger = logging.getLogger(__name__)

def load_lfp_tensor(tensor_path: Path) -> tuple[NDArray[np.float64], float, list[str], float, float]:
    if not tensor_path.exists(): raise FileNotFoundError(f"LFP tensor not found: {tensor_path}")
    # FIX: Security risk removed
    npz = np.load(tensor_path, allow_pickle=False)
    data = npz["data"]
    sfreq = float(npz["sfreq"])
    roi_names = [str(r) for r in npz["roi_names"]]
    tmin = float(npz["tmin"])
    tmax = float(npz["tmax"])
    if data.ndim != 4: raise ValueError(f"Expected 4D tensor, got shape {data.shape}")
    return data, sfreq, roi_names, tmin, tmax

def get_plot_label(channel_name: str, labels_map: dict[str, str]) -> str:
    return labels_map.get(channel_name, channel_name)

def compute_roi_tfr(epochs_data: NDArray[np.float64], roi_name: str, sfreq: float, tmin: float, freqs: NDArray[np.float64], n_cycles: NDArray[np.float64]) -> mne.time_frequency.AverageTFR:
    # FIX: Use 'seeg' for proper intracranial scaling in MNE
    info = mne.create_info(ch_names=[roi_name], sfreq=sfreq, ch_types=["seeg"])
    epochs = mne.EpochsArray(epochs_data[:, np.newaxis, :], info, tmin=tmin, verbose=False)
    tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True, verbose=False)
    return tfr

def plot_tfr(tfr: mne.time_frequency.AverageTFR, roi_name: str, plot_label: str, config: TFRConfig, out_dir: Path) -> None:
    fig = tfr.plot(
        picks=roi_name,
        title=f"{config.subject_id} {config.condition} — {plot_label} ({roi_name})",
        show=False, tmin=config.plot_tmin, tmax=config.plot_tmax,
        fmin=config.plot_fmin, fmax=config.plot_fmax,
        baseline=config.baseline_window, mode="percent", vlim=config.vlim,
    )
    if config.save_plots:
        # tfr.plot returns a list of figures or a single figure depending on picks
        if isinstance(fig, list): fig = fig[0]
        fig.savefig(out_dir / f"TFR_{roi_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    if config.show_plots:
        plt.show()

def generate_tfr_plots(config: TFRConfig) -> None:
    data, sfreq, roi_names, tensor_tmin, tensor_tmax = load_lfp_tensor(config.tensor_file)
    n_subjects, n_rois, n_epochs, n_times = data.shape
    if n_subjects != 1: raise ValueError(f"Expected 1 subject, got {n_subjects}")

    subject_dir = config.output_base / config.subject_id / config.condition
    subject_dir.mkdir(parents=True, exist_ok=True)

    freqs_array = np.array(config.freqs)
    if freqs_array.max() >= sfreq / 2:
        raise ValueError(f"Max frequency {freqs_array.max()} exceeds Nyquist ({sfreq/2})")
        
    n_cycles = freqs_array / config.n_cycles_divisor

    for roi_idx, roi_name in enumerate(roi_names):
        plot_label = get_plot_label(roi_name, config.anatomical_labels)
        logger.info("Processing ROI [%d/%d]: %s", roi_idx + 1, n_rois, plot_label)

        roi_epochs = data[0, roi_idx, :, :]
        tfr = compute_roi_tfr(roi_epochs, roi_name, sfreq, tensor_tmin, freqs_array, n_cycles)
        
        # FIX: Crop TFR to desired window AFTER wavelet convolution to prevent baseline inflation
        tfr.crop(tmin=config.plot_tmin, tmax=config.plot_tmax)

        out_file = subject_dir / f"TFR_{roi_name}.h5"
        tfr.save(out_file, overwrite=True)
        plot_tfr(tfr, roi_name, plot_label, config, subject_dir)

    logger.info("All TFRs saved to %s", subject_dir)

