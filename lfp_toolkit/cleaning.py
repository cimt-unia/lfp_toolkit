# lfp_toolkit/cleaning.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import mne
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, welch

from .io import load_raw
from .config import CleaningConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Types
# ---------------------------------------------------------------------------
_ART_WIDTH_SEARCH_SEC: float = 0.04
_ART_TIME_B4_PEAK_SEC: float = 0.06
_ART_TIME_AFTER_PEAK_SEC: float = 0.30
_TWAVE_TIME_SEC: float = 0.15
_SHIFT_SEARCH_RANGE: int = 4

class LFPCleanResult(TypedDict, total=False):
    data: NDArray[np.float64]
    roi_names: list[str]
    subject_id: str
    sfreq: float
    raw_filtered: NDArray[np.float64]
    removed: NDArray[np.float64]
    spike_mask: NDArray[np.bool_]
    dead_mask: NDArray[np.bool_]
    reconstructed_mask: NDArray[np.bool_]
    ekg_peaks: list[list[int]]
    ekg_templates: list[NDArray[np.float64] | None]
    ekg_counts: list[int]
    ekg_thresholds: list[float]

@dataclass(frozen=True)
class _EKGParams:
    fs: float
    thr_pos: float
    thr_neg: float
    art_width_search_pts: int
    art_pts_b4_peak: int
    art_pts_after_peak: int
    twave_pts: int

    @property
    def art_pts_plus_t_after_peak(self) -> int:
        return self.art_pts_after_peak + self.twave_pts

    @property
    def art_width_pts(self) -> int:
        return self.art_pts_b4_peak + self.art_pts_plus_t_after_peak + 1

    @property
    def qrs_width(self) -> int:
        return self.art_pts_b4_peak + self.art_pts_after_peak + 1

# ---------------------------------------------------------------------------
# Numerical Helpers
# ---------------------------------------------------------------------------
def _matlab_round(x: float) -> int:
    return int(np.floor(x + 0.5))

def _compute_robust_stats(sig: NDArray[np.float64]) -> tuple[float, float]:
    finite = sig[np.isfinite(sig)]
    if finite.size == 0:
        return 0.0, 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    robust_std = mad * 1.4826
    return med, robust_std

def _resolve_two_sided_threshold(
    sig: NDArray[np.float64], thr: float | str | None, sensitivity: float,
) -> tuple[float, float]:
    if isinstance(thr, str):
        if thr != "auto":
            raise ValueError("String threshold must be 'auto'.")
    elif thr is not None and not isinstance(thr, str):
        return float(thr), -float(thr)

    med, robust_std = _compute_robust_stats(sig)
    if robust_std > 0.0:
        return med + sensitivity * robust_std, med - sensitivity * robust_std

    p99 = float(np.percentile(sig[np.isfinite(sig)], 99.0))
    p01 = float(np.percentile(sig[np.isfinite(sig)], 1.0))
    return p99, p01

def _compute_ekg_params(fs: float, thr_pos: float, thr_neg: float) -> _EKGParams:
    return _EKGParams(
        fs=fs, thr_pos=thr_pos, thr_neg=thr_neg,
        art_width_search_pts=max(1, _matlab_round(_ART_WIDTH_SEARCH_SEC * fs)),
        art_pts_b4_peak=max(1, _matlab_round(_ART_TIME_B4_PEAK_SEC * fs)),
        art_pts_after_peak=max(1, _matlab_round(_ART_TIME_AFTER_PEAK_SEC * fs)),
        twave_pts=max(1, _matlab_round(_TWAVE_TIME_SEC * fs)),
    )

def _highpass_template(template: NDArray[np.float64], fs: float, cutoff_hz: float) -> NDArray[np.float64]:
    nyq = fs / 2.0
    if cutoff_hz >= nyq or len(template) < 10:
        return template.copy()
    b, a = butter(3, cutoff_hz / nyq, btype="high")
    padlen = min(3 * max(len(a), len(b)), len(template) - 1)
    if padlen < 1:
        return template.copy()
    return filtfilt(b, a, template, padlen=padlen)

def _detect_dead_segments(
    sig: NDArray[np.float64], sfreq: float, window_sec: float = 0.2,
    std_ratio: float = 0.05, min_gap_merge_sec: float = 1.0, min_dead_duration_sec: float = 0.3,
) -> NDArray[np.bool_]:
    n = len(sig)
    if n < 2:
        return np.zeros(n, dtype=bool)

    window_len = max(1, int(sfreq * window_sec))
    n_windows = int(np.ceil(n / window_len))
    window_stds = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * window_len
        end = min(start + window_len, n)
        seg = sig[start:end]
        finite = seg[np.isfinite(seg)]
        window_stds[i] = float(np.std(finite)) if finite.size > 1 else 0.0

    valid_stds = window_stds[window_stds > 0]
    if valid_stds.size == 0:
        return np.ones(n, dtype=bool)

    median_std = float(np.median(valid_stds))
    threshold = median_std * std_ratio
    raw_mask = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        if window_stds[i] <= threshold:
            raw_mask[i] = True

    sample_mask = np.zeros(n, dtype=bool)
    for i in range(n_windows):
        if raw_mask[i]:
            start = i * window_len
            end = min(start + window_len, n)
            sample_mask[start:end] = True

    diff = np.diff(sample_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if sample_mask[0]: starts = np.concatenate([[0], starts])
    if sample_mask[-1]: ends = np.concatenate([ends, [n]])

    merge_samples = int(min_gap_merge_sec * sfreq)
    merged_starts, merged_ends = [], []
    if len(starts) > 0:
        current_start, current_end = starts[0], ends[0]
        for s, e in zip(starts[1:], ends[1:]):
            if s - current_end <= merge_samples:
                current_end = e
            else:
                merged_starts.append(current_start)
                merged_ends.append(current_end)
                current_start, current_end = s, e
        merged_starts.append(current_start)
        merged_ends.append(current_end)

    min_dead_samples = int(min_dead_duration_sec * sfreq)
    final_mask = np.zeros(n, dtype=bool)
    for s, e in zip(merged_starts, merged_ends):
        if (e - s) >= min_dead_samples:
            final_mask[s:e] = True
    return final_mask

def _reconstruct_dead_segments(
    sig: NDArray[np.float64], dead_mask: NDArray[np.bool_], sfreq: float, context_sec: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    out = sig.copy()
    recon_mask = np.zeros_like(dead_mask)
    if not dead_mask.any():
        return out, recon_mask

    diff = np.diff(dead_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if dead_mask[0]: starts = np.concatenate([[0], starts])
    if dead_mask[-1]: ends = np.concatenate([ends, [len(dead_mask)]])

    context_samples = int(context_sec * sfreq)
    for start, end in zip(starts, ends):
        length = end - start
        if length == 0: continue

        ctx_before_start = max(0, start - context_samples)
        ctx_before = sig[ctx_before_start:start]
        ctx_after_end = min(len(sig), end + context_samples)
        ctx_after = sig[end:ctx_after_end]

        context_data = np.concatenate([ctx_before, ctx_after])
        if len(context_data) < 10:
            out[start:end] = 0.0
            recon_mask[start:end] = True
            continue

        nperseg = min(len(context_data), int(sfreq * 1.0))
        freqs, psd = welch(context_data, fs=sfreq, nperseg=nperseg, window="hann")
        phases = np.random.uniform(0, 2 * np.pi, len(freqs))
        amplitude = np.sqrt(psd * sfreq * nperseg / 2)
        spectrum = amplitude * np.exp(1j * phases)
        surrogate = np.fft.irfft(spectrum, n=length)

        context_rms = np.sqrt(np.mean(context_data ** 2))
        surrogate_rms = np.sqrt(np.mean(surrogate ** 2))
        if surrogate_rms > 0:
            surrogate *= context_rms / surrogate_rms

        fade_len = min(20, length // 4)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            surrogate[:fade_len] *= fade_in
            if length > fade_len:
                surrogate[-fade_len:] *= fade_out

        out[start:end] = surrogate
        recon_mask[start:end] = True
    return out, recon_mask

def _detect_ekg_peaks(sig: NDArray[np.float64], params: _EKGParams) -> list[int]:
    n = len(sig)
    peaks: list[int] = []
    if n < 2: return peaks

    prev_pt = sig[0]
    data_index = 1
    while True:
        if data_index >= n: break
        found_threshold_crossing = False
        if prev_pt <= params.thr_pos and sig[data_index] > params.thr_pos:
            found_threshold_crossing = True

        if found_threshold_crossing:
            if data_index + params.art_width_search_pts >= n: break
            peak_dat = sig[data_index]
            peak_index = data_index
            for k in range(1, params.art_width_search_pts):
                idx = data_index + k
                if sig[idx] > peak_dat:
                    peak_dat = sig[idx]
                    peak_index = idx
            peaks.append(int(peak_index))
            data_index = peak_index + params.art_pts_plus_t_after_peak
            if data_index >= n - 1: break
            prev_pt = sig[data_index - 1]
        else:
            prev_pt = sig[data_index]
            data_index += 1
    return peaks

def _fit_scale_fixed_offset(template: NDArray[np.float64], segment: NDArray[np.float64], offset: float) -> float:
    assert len(template) == len(segment)
    y = segment - offset
    denom = float(np.dot(template, template))
    if denom <= 0.0: return 0.0
    return float(np.dot(template, y) / denom)

def _ekg_artifact_removal(
    sig: NDArray[np.float64], fs: float, thr: float | str | None, polarity: int,
    auto_sensitivity: float, subtract_t_wave: bool = False, qrs_highpass_hz: float = 4.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int], NDArray[np.float64] | None, float]:
    sig = np.asarray(sig, dtype=np.float64).ravel()
    n = len(sig)
    if n == 0: return sig.copy(), np.zeros_like(sig), [], None, 0.0

    if not np.all(np.isfinite(sig)):
        fill = float(np.nanmedian(sig[np.isfinite(sig)])) if np.any(np.isfinite(sig)) else 0.0
        sig = np.nan_to_num(sig, nan=fill, posinf=fill, neginf=fill)

    if polarity not in (-1, 1): raise ValueError("polarity must be +1 or -1.")
    detect_sig = -sig if polarity == -1 else sig
    thr_pos, thr_neg = _resolve_two_sided_threshold(detect_sig, thr, auto_sensitivity)
    params = _compute_ekg_params(fs, thr_pos, thr_neg)
    peaks = _detect_ekg_peaks(detect_sig, params)

    if not peaks:
        return sig.copy(), np.zeros_like(sig), peaks, None, thr_pos

    art_sum = np.zeros(params.art_width_pts, dtype=np.float64)
    art_avg_count = 0
    for pk in peaks:
        start = pk - params.art_pts_b4_peak
        end = pk + params.art_pts_plus_t_after_peak + 1
        if 0 <= start and end <= n:
            art_sum += sig[start:end]
            art_avg_count += 1

    if art_avg_count == 0:
        return sig.copy(), np.zeros_like(sig), peaks, None, thr_pos

    qrst = art_sum / float(art_avg_count)
    sigin = sig.copy()
    sigout = sigin.copy()
    mn = float(np.min(qrst))
    qrst = qrst - mn

    if np.allclose(qrst, 0.0):
        return sigout, np.zeros_like(sig), peaks, None, thr_pos

    qrs_width = params.qrs_width
    t_width = len(qrst) - qrs_width
    if t_width < 0: raise ValueError("Widths do not match QRST length.")

    qrs_model_raw = qrst[:qrs_width]
    qrs_model = _highpass_template(qrs_model_raw, fs, cutoff_hz=qrs_highpass_hz)
    pre = params.art_pts_b4_peak
    post = params.art_pts_after_peak

    for pk in peaks:
        qrs_loc1 = pk + 1
        if not (qrs_loc1 > pre + _SHIFT_SEARCH_RANGE and qrs_loc1 + post + _SHIFT_SEARCH_RANGE < n):
            continue

        best_qrs_err = np.inf
        qrs_scale = 1.0
        qrs_shift = 0
        qrs_ok = False

        for shift in range(-_SHIFT_SEARCH_RANGE, _SHIFT_SEARCH_RANGE + 1):
            start = pk - pre + shift
            end = pk + post + shift + 1
            if start < 0 or end > n: continue
            match_qrs = sigin[start:end]
            if len(match_qrs) != len(qrs_model) or not np.all(np.isfinite(match_qrs)): continue
            scale = _fit_scale_fixed_offset(qrs_model, match_qrs, offset=0.0)
            err = float(np.sum((match_qrs - scale * qrs_model) ** 2))
            if shift == -_SHIFT_SEARCH_RANGE or err < best_qrs_err:
                best_qrs_err = err
                qrs_scale = scale
                qrs_shift = shift
                qrs_ok = True

        if not qrs_ok: continue

        qrs_start = pk - pre + qrs_shift
        qrs_end = qrs_start + len(qrs_model)
        if 0 <= qrs_start and qrs_end <= n:
            sigout[qrs_start:qrs_end] -= qrs_scale * qrs_model

        if subtract_t_wave and t_width > 0:
            t_model = qrst[qrs_width:]
            best_t_err = np.inf
            t_scale = 1.0
            t_shift = 0
            t_ok = False

            for shift in range(-_SHIFT_SEARCH_RANGE, _SHIFT_SEARCH_RANGE + 1):
                start = pk + post + 1 + shift
                end = pk + post + t_width + shift + 1
                if start < 0 or end > n: continue
                match_t = sigin[start:end]
                if len(match_t) != len(t_model) or not np.all(np.isfinite(match_t)): continue
                scale = _fit_scale_fixed_offset(t_model, match_t, offset=0.0)
                err = float(np.sum((match_t - scale * t_model) ** 2))
                if shift == -_SHIFT_SEARCH_RANGE or err < best_t_err:
                    best_t_err = err
                    t_scale = scale
                    t_shift = shift
                    t_ok = True

            if t_ok:
                overlap = max(0, qrs_shift - t_shift)
                if overlap < len(t_model):
                    t_trimmed = t_model[overlap:]
                    t_start = pk + post + 1 + t_shift + overlap
                    t_end = t_start + len(t_trimmed)
                    if 0 <= t_start and t_end <= n:
                        sigout[t_start:t_end] -= t_scale * t_trimmed

    removed = sigin - sigout
    return sigout, removed, peaks, qrst, thr_pos

def _soft_notch(data: NDArray[np.float64], sfreq: float, freq: float, max_db: float, sigma_hz: float) -> NDArray[np.float64]:
    squeezed = False
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
        squeezed = True

    n_times = arr.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    max_gain = 10.0 ** (-max_db / 20.0)
    attenuation = 1.0 - (1.0 - max_gain) * np.exp(-0.5 * ((freqs - freq) / sigma_hz) ** 2)

    out = np.zeros_like(arr)
    for ch in range(arr.shape[0]):
        spectrum = np.fft.rfft(arr[ch])
        out[ch] = np.fft.irfft(spectrum * attenuation, n=n_times)
    return out.squeeze(axis=0) if squeezed else out

def _detect_spikes(sig: NDArray[np.float64], sfreq: float, segment_sec: float, mad_threshold: float) -> NDArray[np.bool_]:
    n = len(sig)
    segment_len = max(1, int(sfreq * segment_sec))
    n_segments = int(np.ceil(n / segment_len))
    mask = np.zeros(n, dtype=bool)
    for i in range(n_segments):
        start = i * segment_len
        end = min(start + segment_len, n)
        segment = sig[start:end]
        finite = segment[np.isfinite(segment)]
        if finite.size < 3: continue
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        if mad <= 0.0: continue
        robust_std = mad * 1.4826
        upper = med + mad_threshold * robust_std
        lower = med - mad_threshold * robust_std
        mask[start:end] = (segment > upper) | (segment < lower)
    return mask

def _apply_spike_mask(sig: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    out = sig.copy()
    out[mask] = np.nan
    return out

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean_lfp(
    raw_data: mne.io.Raw | str | Path,
    *,
    file_format: str | None = None,
    subject_id: str | None = None,
    config: CleaningConfig | None = None,
    return_removed: bool = False,
    return_raw: bool = False,
    spike_removal: bool = False,
    detect_spikes: bool = True,
    detect_dead_segments: bool = True,
    reconstruct_dead: bool | None = None,
) -> LFPCleanResult:
    """
    Clean LFP data with Delta-safe QRS-only EKG subtraction.
    Accepts an mne.io.Raw object or a file path.
    """
    cfg = config or CleaningConfig()
    if reconstruct_dead is None:
        reconstruct_dead = cfg.reconstruct_dead

    if isinstance(raw_data, (str, Path)):
        raw = load_raw(raw_data, file_format=file_format, preload=True, verbose="WARNING")
        path = Path(raw_data)
        if subject_id is None:
            subject_id = path.stem.split("_")[0] if path.stem else "unknown"
    else:
        raw = raw_data
        if subject_id is None:
            subject_id = "unknown"

    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        picks = mne.pick_types(raw.info, meg=False, eeg=False, misc=True, seeg=True, dbs=True)
    if len(picks) == 0:
        raise ValueError("No valid data channels found.")

    roi_names = [raw.ch_names[p] for p in picks]
    sfreq = float(raw.info["sfreq"])

    if cfg.bandpass_l_freq is not None or cfg.bandpass_h_freq is not None:
        raw.filter(l_freq=cfg.bandpass_l_freq, h_freq=cfg.bandpass_h_freq, picks=picks, verbose=False)

    eeg = raw.get_data(picks=picks)

    if cfg.notch_freq is not None and cfg.notch_freq > 0.0:
        notched = _soft_notch(eeg, sfreq, cfg.notch_freq, cfg.notch_max_db, cfg.notch_sigma_hz)
    else:
        notched = eeg.copy()

    n_channels, n_times = notched.shape
    cleaned = np.zeros_like(notched)
    removed = np.zeros_like(notched)
    spike_masks = np.zeros((n_channels, n_times), dtype=bool)
    dead_masks = np.zeros((n_channels, n_times), dtype=bool)
    recon_masks = np.zeros((n_channels, n_times), dtype=bool)

    ekg_peaks: list[list[int]] = []
    ekg_templates: list[NDArray[np.float64] | None] = []
    ekg_counts: list[int] = []
    ekg_thresholds: list[float] = []

    for i, name in enumerate(roi_names):
        if detect_dead_segments:
            dead_masks[i] = _detect_dead_segments(
                notched[i], sfreq, window_sec=cfg.dead_segment_sec, std_ratio=cfg.dead_std_ratio,
                min_gap_merge_sec=cfg.min_gap_merge_sec, min_dead_duration_sec=cfg.min_dead_duration_sec,
            )
            if reconstruct_dead and dead_masks[i].any():
                reconstructed_sig, recon_mask = _reconstruct_dead_segments(
                    notched[i], dead_masks[i], sfreq, context_sec=cfg.recon_context_sec
                )
                recon_masks[i] = recon_mask
                input_for_ekg = reconstructed_sig
            else:
                input_for_ekg = notched[i].copy()
        else:
            input_for_ekg = notched[i].copy()

        cleaned_ekg, removed_ch, peaks_ch, template_ch, thr_used = _ekg_artifact_removal(
            input_for_ekg, fs=sfreq, thr=cfg.ekg_thr, polarity=cfg.ekg_polarity,
            auto_sensitivity=cfg.ekg_auto_sensitivity, subtract_t_wave=cfg.subtract_t_wave,
            qrs_highpass_hz=cfg.qrs_highpass_hz,
        )

        ekg_peaks.append(peaks_ch)
        ekg_templates.append(template_ch)
        ekg_counts.append(len(peaks_ch))
        ekg_thresholds.append(thr_used)

        if detect_spikes:
            mask_ch = _detect_spikes(cleaned_ekg, sfreq, cfg.spike_segment_sec, cfg.spike_mad_thresh)
            spike_masks[i] = mask_ch
            cleaned[i] = _apply_spike_mask(cleaned_ekg, mask_ch) if spike_removal else cleaned_ekg
        else:
            cleaned[i] = cleaned_ekg

        if return_removed:
            removed[i] = removed_ch

    result: LFPCleanResult = {
        "data": cleaned[np.newaxis, :, :],
        "roi_names": roi_names,
        "subject_id": subject_id,
        "sfreq": sfreq,
        "ekg_peaks": ekg_peaks,
        "ekg_templates": ekg_templates,
        "ekg_counts": ekg_counts,
        "ekg_thresholds": ekg_thresholds,
    }

    if return_raw: result["raw_filtered"] = eeg[np.newaxis, :, :]
    if return_removed: result["removed"] = removed[np.newaxis, :, :]
    if detect_spikes: result["spike_mask"] = spike_masks[np.newaxis, :, :]
    if detect_dead_segments: result["dead_mask"] = dead_masks[np.newaxis, :, :]
    if reconstruct_dead: result["reconstructed_mask"] = recon_masks[np.newaxis, :, :]

    return result