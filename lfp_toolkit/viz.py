# lfp_toolkit/viz.py

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FREQ_MAX_HZ = 50.0
PSD_WINDOW_SEC = 4.0
PSD_OVERLAP = 0.75
PSD_EPS = 1e-15

RAW_COLOR = "#888888"
CLEANED_COLOR = "#1F77B4"
REMOVED_COLOR = "#D62728"
DEAD_COLOR = "#FF7F0E"
SPIKE_COLOR = "#9467BD"

PLOT_BANDS = [
    (0.5, 4.0, "Delta", "#90B3F9"),
    (4.0, 8.0, "Theta", "#FFF9B2"),
    (8.0, 13.0, "Alpha", "#AAFCD2"),
    (13.0, 20.0, "Low Beta", "#97C2F9"),
    (20.0, 30.0, "High Beta", "#90BEF5"),
]

def _masked(sig: NDArray[np.float64]) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(sig)

def _fill_nan(sig: NDArray[np.float64]) -> NDArray[np.float64]:
    sig = np.asarray(sig, dtype=np.float64)
    finite = np.isfinite(sig)
    if not finite.any(): return np.zeros_like(sig)
    if finite.all(): return sig
    fill = float(np.nanmedian(sig[finite]))
    return np.where(finite, sig, fill)

def _psd_db(sig: NDArray[np.float64], sfreq: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sig = _fill_nan(sig)
    if len(sig) < 16: return np.array([]), np.array([])
    nperseg = min(int(sfreq * PSD_WINDOW_SEC), len(sig))
    noverlap = int(nperseg * PSD_OVERLAP)
    freqs, psd = sp_signal.welch(sig, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window="hann")
    mask = freqs <= FREQ_MAX_HZ
    return freqs[mask], 10.0 * np.log10(psd[mask] + PSD_EPS)

def _delta_ratio(raw_sig: NDArray[np.float64], cleaned_sig: NDArray[np.float64], sfreq: float) -> float:
    r = _fill_nan(raw_sig)
    c = _fill_nan(cleaned_sig)
    nperseg = min(int(sfreq * PSD_WINDOW_SEC), len(r))
    noverlap = int(nperseg * PSD_OVERLAP)
    freqs, psd_r = sp_signal.welch(r, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window="hann")
    _, psd_c = sp_signal.welch(c, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window="hann")
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    return float(np.mean(psd_c[delta_mask]) / (np.mean(psd_r[delta_mask]) + PSD_EPS))

def _add_band_shading(ax: plt.Axes) -> None:
    for f_lo, f_hi, name, color in PLOT_BANDS:
        if f_hi <= FREQ_MAX_HZ:
            ax.axvspan(f_lo, f_hi, facecolor=color, alpha=0.08, edgecolor="none")
            ax.annotate(name, xy=((f_lo + f_hi) / 2.0, 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8, color="#1E3A5F", fontweight="bold", alpha=0.8)

def viz_data(
    result: dict, zoom_sec: float = 5.0, max_channels: int | None = None,
    save_dir: str | Path | None = None, show: bool = True,
) -> None:
    """Visualize lfp_xkit cleaning results with dead/spike overlays."""
    data = np.asarray(result["data"])[0]
    roi_names = list(result["roi_names"])
    sfreq = float(result["sfreq"])

    raw = result.get("raw_filtered")
    if raw is not None: raw = np.asarray(raw)[0]
    removed = result.get("removed")
    if removed is not None: removed = np.asarray(removed)[0]
    spike_mask = result.get("spike_mask")
    if spike_mask is not None: spike_mask = np.asarray(spike_mask)[0]
    dead_mask = result.get("dead_mask")
    if dead_mask is not None: dead_mask = np.asarray(dead_mask)[0]

    ekg_peaks = result.get("ekg_peaks", [])
    ekg_templates = result.get("ekg_templates", [])
    ekg_counts = result.get("ekg_counts", [])
    ekg_thresholds = result.get("ekg_thresholds", [])

    n_channels = len(roi_names)
    if max_channels is not None: n_channels = min(n_channels, int(max_channels))
    times = np.arange(data.shape[-1]) / sfreq

    for i in range(n_channels):
        name = roi_names[i]
        count = ekg_counts[i] if i < len(ekg_counts) else 0
        thr_used = ekg_thresholds[i] if i < len(ekg_thresholds) else np.nan
        peaks = ekg_peaks[i] if i < len(ekg_peaks) else []
        template = ekg_templates[i] if i < len(ekg_templates) else None

        n_rows = 0
        if raw is not None: n_rows += 1
        n_rows += 1
        if removed is not None: n_rows += 1

        fig_full, axes_full = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), sharex=True)
        if n_rows == 1: axes_full = [axes_full]

        ax_idx = 0
        if raw is not None:
            axes_full[ax_idx].plot(times, _masked(raw[i]), linewidth=0.3, color=RAW_COLOR)
            if dead_mask is not None:
                dead_times = times[dead_mask[i]]
                axes_full[ax_idx].scatter(dead_times, raw[i][dead_mask[i]], c=DEAD_COLOR, s=10, zorder=5, label="Dead")
            axes_full[ax_idx].set_ylabel("Raw"); axes_full[ax_idx].grid(True, alpha=0.25, ls="--")
            ax_idx += 1

        axes_full[ax_idx].plot(times, _masked(data[i]), linewidth=0.3, color=CLEANED_COLOR)
        if dead_mask is not None:
            dead_times = times[dead_mask[i]]
            axes_full[ax_idx].scatter(dead_times, data[i][dead_mask[i]], c=DEAD_COLOR, s=10, zorder=5, label="Dead")
        if spike_mask is not None:
            spike_times = times[spike_mask[i]]
            axes_full[ax_idx].scatter(spike_times, data[i][spike_mask[i]], c=SPIKE_COLOR, s=10, zorder=5, label="Spikes")
        axes_full[ax_idx].set_ylabel("Cleaned"); axes_full[ax_idx].grid(True, alpha=0.25, ls="--")
        ax_idx += 1

        if removed is not None:
            axes_full[ax_idx].plot(times, _masked(removed[i]), linewidth=0.3, color=REMOVED_COLOR)
            axes_full[ax_idx].set_ylabel("Removed"); axes_full[ax_idx].grid(True, alpha=0.25, ls="--")

        axes_full[-1].set_xlabel("Time (s)")
        fig_full.suptitle(f"{name} — Full Segment", fontsize=12, fontweight="bold")
        fig_full.tight_layout()
        if save_dir: fig_full.savefig(Path(save_dir) / f"{name}_full.png", dpi=150, bbox_inches="tight")

        zoom_samples = int(zoom_sec * sfreq)
        start = 0
        valid_peak = next((pk for pk in peaks if pk > int(0.5 * sfreq) and pk + zoom_samples < len(times)), None)
        if valid_peak is not None:
            start = max(0, min(valid_peak - int(0.5 * sfreq), len(times) - zoom_samples))
        end = min(len(times), start + zoom_samples)
        zs = slice(start, end)
        t_z = times[zs]

        fig_zoom, axes_zoom = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), sharex=True)
        if n_rows == 1: axes_zoom = [axes_zoom]

        ax_idx = 0
        if raw is not None:
            axes_zoom[ax_idx].plot(t_z, _masked(raw[i, zs]), linewidth=0.7, color=RAW_COLOR)
            for pk in peaks:
                if start <= pk < end: axes_zoom[ax_idx].axvline(times[pk], color=REMOVED_COLOR, alpha=0.4, lw=0.8)
            axes_zoom[ax_idx].set_ylabel("Raw"); axes_zoom[ax_idx].grid(True, alpha=0.25, ls="--")
            ax_idx += 1

        axes_zoom[ax_idx].plot(t_z, _masked(data[i, zs]), linewidth=0.7, color=CLEANED_COLOR)
        for pk in peaks:
            if start <= pk < end: axes_zoom[ax_idx].axvline(times[pk], color=REMOVED_COLOR, alpha=0.4, lw=0.8)
        axes_zoom[ax_idx].set_ylabel("Cleaned"); axes_zoom[ax_idx].grid(True, alpha=0.25, ls="--")
        ax_idx += 1

        if removed is not None:
            axes_zoom[ax_idx].plot(t_z, _masked(removed[i, zs]), linewidth=0.7, color=REMOVED_COLOR)
            axes_zoom[ax_idx].set_ylabel("Removed"); axes_zoom[ax_idx].grid(True, alpha=0.25, ls="--")

        axes_zoom[-1].set_xlabel("Time (s)")
        fig_zoom.suptitle(f"{name} — Zoom ({zoom_sec}s)", fontsize=12, fontweight="bold")
        fig_zoom.tight_layout()
        if save_dir: fig_zoom.savefig(Path(save_dir) / f"{name}_zoom.png", dpi=150, bbox_inches="tight")

        if raw is not None:
            fig_psd, axes_psd = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fr, pr = _psd_db(raw[i], sfreq)
            fc, pc = _psd_db(data[i], sfreq)
            _add_band_shading(axes_psd[0])
            axes_psd[0].plot(fr, pr, color=RAW_COLOR, lw=1.5); axes_psd[0].set_ylabel("Raw PSD (dB/Hz)")
            axes_psd[0].grid(True, alpha=0.25, ls="--")
            _add_band_shading(axes_psd[1])
            axes_psd[1].plot(fc, pc, color=CLEANED_COLOR, lw=1.5); axes_psd[1].set_ylabel("Cleaned PSD (dB/Hz)")
            axes_psd[1].set_xlabel("Frequency (Hz)"); axes_psd[1].grid(True, alpha=0.25, ls="--")
        else:
            fig_psd, ax_psd = plt.subplots(figsize=(12, 4))
            fc, pc = _psd_db(data[i], sfreq)
            _add_band_shading(ax_psd)
            ax_psd.plot(fc, pc, color=CLEANED_COLOR, lw=1.5)
            ax_psd.set_ylabel("Cleaned PSD (dB/Hz)"); ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.grid(True, alpha=0.25, ls="--")

        fig_psd.suptitle(f"{name} — PSD", fontsize=12, fontweight="bold")
        fig_psd.tight_layout()
        if save_dir: fig_psd.savefig(Path(save_dir) / f"{name}_psd.png", dpi=150, bbox_inches="tight")

        if template is not None:
            fig_tmpl, ax_tmpl = plt.subplots(figsize=(10, 3))
            tmpl_t = np.arange(len(template)) / sfreq
            ax_tmpl.plot(tmpl_t, template, color="#111111", lw=1.5)
            ax_tmpl.set_xlabel("Time from fiducial (s)"); ax_tmpl.set_ylabel("Amplitude")
            ax_tmpl.set_title(f"{name} — QRST Template", fontsize=11, fontweight="bold")
            ax_tmpl.grid(True, alpha=0.25, ls="--")
            fig_tmpl.tight_layout()
            if save_dir: fig_tmpl.savefig(Path(save_dir) / f"{name}_template.png", dpi=150, bbox_inches="tight")

    if show and save_dir is None:
        plt.show(block=True)