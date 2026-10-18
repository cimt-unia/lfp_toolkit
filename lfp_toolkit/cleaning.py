# lfp_toolkit/cleaning.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, Any

import mne
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt, welch

from .io import load_raw
from .config import CleaningConfig

logger = logging.getLogger(__name__)

_ART_WIDTH_SEARCH_SEC: float = 0.04
_ART_TIME_B4_PEAK_SEC: float = 0.06
_ART_TIME_AFTER_PEAK_SEC: float = 0.30
_TWAVE_TIME_SEC: float = 0.15
_SHIFT_SEARCH_RANGE: int = 4

DELTA_RATIO_GOOD_MIN = 0.85
DELTA_RATIO_WARN_MIN = 0.70
NAN_PCT_WARN = 5.0
NAN_PCT_REJECT = 30.0

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
    nan_mask: NDArray[np.bool_]
    ekg_peaks: list[list[int]]
    ekg_templates: list[NDArray[np.float64] | None]
    ekg_counts: list[int]
    ekg_thresholds: list[float]
    channel_qc: list[dict[str, Any]]

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

def _matlab_round(x: float) -> int:
    return int(np.floor(x + 0.5))

def _compute_robust_stats(sig: NDArray[np.float64]) -> tuple[float, float]:
    finite = sig[np.isfinite(sig)]
    if finite.size == 0: return 0.0, 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return med, mad * 1.4826

def _resolve_two_sided_threshold(sig: NDArray[np.float64], thr: float | str | None, sensitivity: float) -> tuple[float, float]:
    if isinstance(thr, str):
        if thr != "auto": raise ValueError("String threshold must be 'auto'.")
    elif thr is not None:
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

def _resolve_picks(raw: mne.io.BaseRaw, picks: str | list[str]) -> NDArray[np.int64]:
    if picks == "eeg":
        idx = np.asarray(mne.pick_types(raw.info, eeg=True), dtype=np.int64)
    elif picks == "all":
        idx = np.arange(len(raw.ch_names), dtype=np.int64)
    elif isinstance(picks, (list, tuple)):
        wanted = set(picks)
        idx = np.asarray([i for i, t in enumerate(raw.get_channel_types()) if t in wanted], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported picks={picks!r}")

    if idx.size == 0:
        raise ValueError(f"No channels selected for picks={picks!r}. Available: {sorted(set(raw.get_channel_types()))}")
    return idx

def _highpass_template(template: NDArray[np.float64], fs: float, cutoff_hz: float) -> NDArray[np.float64]:
    nyq = fs / 2.0
    if cutoff_hz >= nyq or len(template) < 10: return template.copy()
    # FIX: Use 'sos' for numerical stability at high sampling rates
    sos = butter(3, cutoff_hz / nyq, btype="high", output="sos")
    padlen = min(3 * len(sos) * 2, len(template) - 1)
    if padlen < 1: return template.copy()
    return sosfiltfilt(sos, template, padlen=padlen)

def _detect_dead_segments(sig: NDArray[np.float64], sfreq: float, window_sec: float = 0.2, std_ratio: float = 0.05, min_gap_merge_sec: float = 1.0, min_dead_duration_sec: float = 0.3) -> NDArray[np.bool_]:
    n = len(sig)
    if n < 2: return np.zeros(n, dtype=bool)
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
        logger.warning("Signal is completely flat. Marking as dead.")
        return np.ones(n, dtype=bool)

    threshold = float(np.median(valid_stds)) * std_ratio
    raw_mask = window_stds <= threshold
    
    sample_mask = np.repeat(raw_mask, window_len)[:n]

    diff = np.diff(sample_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if sample_mask[0]: starts = np.concatenate([[0], starts])
    if sample_mask[-1]: ends = np.concatenate([ends, [n]])

    merge_samples = int(min_gap_merge_sec * sfreq)
    merged_starts, merged_ends = [], []
    if len(starts) > 0:
        cur_s, cur_e = int(starts[0]), int(ends[0])
        for s, e in zip(starts[1:], ends[1:]):
            if s - cur_e <= merge_samples: cur_e = int(e)
            else:
                merged_starts.append(cur_s); merged_ends.append(cur_e)
                cur_s, cur_e = int(s), int(e)
        merged_starts.append(cur_s); merged_ends.append(cur_e)

    min_dead_samples = int(min_dead_duration_sec * sfreq)
    final_mask = np.zeros(n, dtype=bool)
    for s, e in zip(merged_starts, merged_ends):
        if (e - s) >= min_dead_samples: final_mask[s:e] = True
    return final_mask

def _reconstruct_dead_segments(
    sig: NDArray[np.float64], 
    dead_mask: NDArray[np.bool_], 
    sfreq: float, 
    context_sec: float = 2.0, 
    rng: np.random.Generator | None = None
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Reconstruct dead/flat segments via spectral matching from surrounding healthy context.
    Excludes both NaN values and samples flagged in dead_mask from the context window 
    to prevent biasing the PSD and RMS calculations.
    """
    out = sig.copy()
    recon_mask = np.zeros_like(dead_mask)
    if not dead_mask.any(): 
        return out, recon_mask
        
    if rng is None: 
        rng = np.random.default_rng()

    diff = np.diff(dead_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if dead_mask[0]: 
        starts = np.concatenate([[0], starts])
    if dead_mask[-1]: 
        ends = np.concatenate([ends, [len(dead_mask)]])

    context_samples = int(context_sec * sfreq)
    
    for start, end in zip(starts, ends):
        length = end - start
        if length == 0: 
            continue
            
        # Define context boundaries
        ctx_before_start = max(0, start - context_samples)
        ctx_after_end = min(len(sig), end + context_samples)
        
        ctx_before = sig[ctx_before_start:start]
        ctx_after = sig[end:ctx_after_end]
        
        # Get corresponding mask slices
        mask_before = dead_mask[ctx_before_start:start]
        mask_after = dead_mask[end:ctx_after_end]
        
        # Exclude BOTH dead_mask samples and NaNs from context
        valid_before = ~mask_before & ~np.isnan(ctx_before)
        valid_after = ~mask_after & ~np.isnan(ctx_after)
        
        ctx_before_clean = ctx_before[valid_before]
        ctx_after_clean = ctx_after[valid_after]
        
        context_data = np.concatenate([ctx_before_clean, ctx_after_clean])

        if len(context_data) < 10:
            out[start:end] = 0.0
            recon_mask[start:end] = True
            continue

        nperseg = min(len(context_data), max(8, int(sfreq * 1.0)))
        freqs, psd = welch(context_data, fs=sfreq, nperseg=nperseg, window="hann")
        
        # Interpolate PSD to target frequencies to prevent pitch-shifting
        target_f = np.fft.rfftfreq(length, d=1.0 / sfreq)
        psd_interp = np.interp(target_f, freqs, psd)
        amp = np.sqrt(np.maximum(psd_interp, 0.0) * sfreq * length / 2)
        
        phases = rng.uniform(0, 2 * np.pi, target_f.size)
        phases[0] = 0.0
        if length % 2 == 0: 
            phases[-1] = 0.0  # Nyquist must be real
        
        spectrum = amp * np.exp(1j * phases)
        surrogate = np.fft.irfft(spectrum, n=length)

        # Scale surrogate to match local RMS of healthy context
        context_rms = np.sqrt(np.mean(context_data ** 2))
        surrogate_rms = np.sqrt(np.mean(surrogate ** 2))
        if surrogate_rms > 0: 
            surrogate *= context_rms / surrogate_rms

        # Time-based fade (20ms or 25% of gap length, whichever is smaller)
        fade_len = min(int(0.02 * sfreq), length // 4) 
        if fade_len > 0:
            surrogate[:fade_len] *= np.linspace(0, 1, fade_len)
            if length > fade_len: 
                surrogate[-fade_len:] *= np.linspace(1, 0, fade_len)

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
        found = prev_pt <= params.thr_pos and sig[data_index] > params.thr_pos
        if found:
            if data_index + params.art_width_search_pts >= n: break
            peak_dat = sig[data_index]
            peak_index = data_index
            for k in range(1, params.art_width_search_pts):
                idx = data_index + k
                if sig[idx] > peak_dat: peak_dat = sig[idx]; peak_index = idx
            peaks.append(int(peak_index))
            data_index = peak_index + params.art_pts_plus_t_after_peak
            if data_index >= n - 1: break
            prev_pt = sig[data_index - 1]
        else:
            prev_pt = sig[data_index]
            data_index += 1
    return peaks

def _fit_scale_fixed_offset(template: NDArray[np.float64], segment: NDArray[np.float64], offset: float) -> float:
    y = segment - offset
    denom = float(np.dot(template, template))
    if denom <= 0.0: return 0.0
    return float(np.dot(template, y) / denom)

def _ekg_artifact_removal(sig: NDArray[np.float64], fs: float, thr: float | str | None, polarity: int, auto_sensitivity: float, subtract_t_wave: bool = False, qrs_highpass_hz: float = 4.0) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int], NDArray[np.float64] | None, float]:
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

    if not peaks: return sig.copy(), np.zeros_like(sig), peaks, None, thr_pos

    art_sum = np.zeros(params.art_width_pts, dtype=np.float64)
    art_avg_count = 0
    for pk in peaks:
        start = pk - params.art_pts_b4_peak
        end = pk + params.art_pts_plus_t_after_peak + 1
        if 0 <= start and end <= n:
            art_sum += sig[start:end]
            art_avg_count += 1

    if art_avg_count == 0: return sig.copy(), np.zeros_like(sig), peaks, None, thr_pos

    qrst = art_sum / float(art_avg_count)
    sigin = sig.copy()
    sigout = sigin.copy()
    
    # FIX: Removed baseline shift. Shifting breaks the zero-offset linear fit.
    if np.allclose(qrst, 0.0): return sigout, np.zeros_like(sig), peaks, None, thr_pos

    qrs_width = params.qrs_width
    t_width = len(qrst) - qrs_width
    if t_width < 0: raise ValueError("Widths do not match QRST length.")

    qrs_model = _highpass_template(qrst[:qrs_width], fs, cutoff_hz=qrs_highpass_hz)
    pre = params.art_pts_b4_peak
    post = params.art_pts_after_peak

    for pk in peaks:
        qrs_loc1 = pk + 1
        if not (qrs_loc1 > pre + _SHIFT_SEARCH_RANGE and qrs_loc1 + post + t_width + _SHIFT_SEARCH_RANGE < n): continue

        best_qrs_err = np.inf
        qrs_scale, qrs_shift, qrs_ok = 1.0, 0, False
        for shift in range(-_SHIFT_SEARCH_RANGE, _SHIFT_SEARCH_RANGE + 1):
            s = pk - pre + shift
            e = pk + post + shift + 1
            if s < 0 or e > n: continue
            m = sigin[s:e]
            if len(m) != len(qrs_model) or not np.all(np.isfinite(m)): continue
            scale = _fit_scale_fixed_offset(qrs_model, m, 0.0)
            err = float(np.sum((m - scale * qrs_model) ** 2))
            if shift == -_SHIFT_SEARCH_RANGE or err < best_qrs_err:
                best_qrs_err, qrs_scale, qrs_shift, qrs_ok = err, scale, shift, True
        if not qrs_ok: continue

        qs = pk - pre + qrs_shift
        qe = qs + len(qrs_model)
        if 0 <= qs and qe <= n: sigout[qs:qe] -= qrs_scale * qrs_model

        if subtract_t_wave and t_width > 0:
            t_model = qrst[qrs_width:]
            best_t_err = np.inf
            t_scale, t_shift, t_ok = 1.0, 0, False
            for shift in range(-_SHIFT_SEARCH_RANGE, _SHIFT_SEARCH_RANGE + 1):
                s = pk + post + 1 + shift
                e = pk + post + t_width + shift + 1
                if s < 0 or e > n: continue
                m = sigin[s:e]
                if len(m) != len(t_model) or not np.all(np.isfinite(m)): continue
                scale = _fit_scale_fixed_offset(t_model, m, 0.0)
                err = float(np.sum((m - scale * t_model) ** 2))
                if shift == -_SHIFT_SEARCH_RANGE or err < best_t_err:
                    best_t_err, t_scale, t_shift, t_ok = err, scale, shift, True
            if t_ok:
                overlap = max(0, qrs_shift - t_shift)
                if overlap < len(t_model):
                    tt = t_model[overlap:]
                    ts = pk + post + 1 + t_shift + overlap
                    te = ts + len(tt)
                    if 0 <= ts and te <= n: sigout[ts:te] -= t_scale * tt

    return sigout, sigin - sigout, peaks, qrst, thr_pos

def _soft_notch(data: NDArray[np.float64], sfreq: float, freq: float, max_db: float, sigma_hz: float) -> NDArray[np.float64]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1: arr = arr[np.newaxis, :]
    n_times = arr.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    max_gain = 10.0 ** (-max_db / 20.0)
    att = 1.0 - (1.0 - max_gain) * np.exp(-0.5 * ((freqs - freq) / sigma_hz) ** 2)
    
    # FIX: Vectorized FFT across all channels
    spectrum = np.fft.rfft(arr, axis=-1)
    out = np.fft.irfft(spectrum * att, n=n_times, axis=-1)
    return out.squeeze(axis=0) if data.ndim == 1 else out

def _detect_spikes(sig: NDArray[np.float64], sfreq: float, segment_sec: float, mad_threshold: float) -> NDArray[np.bool_]:
    n = len(sig)
    segment_len = max(1, int(sfreq * segment_sec))
    n_segments = int(np.ceil(n / segment_len))
    mask = np.zeros(n, dtype=bool)
    for i in range(n_segments):
        s = i * segment_len
        e = min(s + segment_len, n)
        seg = sig[s:e]
        finite = seg[np.isfinite(seg)]
        if finite.size < 3: continue
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        if mad <= 0.0: continue
        rstd = mad * 1.4826
        mask[s:e] = (seg > med + mad_threshold * rstd) | (seg < med - mad_threshold * rstd)
    return mask

def _apply_spike_mask(sig: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    out = sig.copy()
    out[mask] = np.nan
    return out

def _compute_delta_ratio(raw_sig: NDArray[np.float64], cleaned_sig: NDArray[np.float64], sfreq: float, l_freq: float | None) -> float:
    if l_freq is not None and l_freq >= 4.0: return np.nan
    r = np.nan_to_num(raw_sig, nan=0.0)
    c = np.nan_to_num(cleaned_sig, nan=0.0)
    if len(r) < 16: return 1.0
    nperseg = min(int(sfreq * 4.0), len(r))
    noverlap = int(nperseg * 0.75)
    freqs, psd_r = welch(r, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window="hann")
    _, psd_c = welch(c, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window="hann")
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    if not delta_mask.any(): return 1.0
    power_r = np.mean(psd_r[delta_mask])
    power_c = np.mean(psd_c[delta_mask])
    if power_r == 0: return 1.0 if power_c == 0 else 0.0
    return float(power_c / power_r)

def _evaluate_channel_quality(raw_sig: NDArray[np.float64], cleaned_sig: NDArray[np.float64], dead_mask: NDArray[np.bool_], spike_mask: NDArray[np.bool_], sfreq: float, ch_name: str, l_freq: float | None) -> dict[str, Any]:
    bad_mask = dead_mask | spike_mask
    artifact_pct = 100.0 * np.sum(bad_mask) / len(bad_mask)
    delta_ratio = _compute_delta_ratio(raw_sig, cleaned_sig, sfreq, l_freq)
    
    status = "GOOD"
    reasons = []
    if artifact_pct >= NAN_PCT_REJECT: status = "REJECT"; reasons.append(f"High artifact load ({artifact_pct:.1f}%)")
    elif artifact_pct >= NAN_PCT_WARN: status = "WARN"; reasons.append(f"Moderate artifact load ({artifact_pct:.1f}%)")
        
    if not np.isnan(delta_ratio):
        if delta_ratio < DELTA_RATIO_WARN_MIN: status = "REJECT"; reasons.append(f"Severe Delta loss ({delta_ratio:.2f})")
        elif delta_ratio < DELTA_RATIO_GOOD_MIN and status == "GOOD": status = "WARN"; reasons.append(f"Moderate Delta loss ({delta_ratio:.2f})")
            
    if status == "GOOD": logger.info("QC [%s]: GOOD | Artifacts: %.1f%% | Delta: %.2f", ch_name, artifact_pct, delta_ratio)
    elif status == "WARN": logger.warning("QC [%s]: WARN | Artifacts: %.1f%% | Delta: %.2f | %s", ch_name, artifact_pct, delta_ratio, "; ".join(reasons))
    else: logger.error("QC [%s]: REJECT | Artifacts: %.1f%% | Delta: %.2f | %s", ch_name, artifact_pct, delta_ratio, "; ".join(reasons))
        
    return {"status": status, "artifact_pct": artifact_pct, "delta_ratio": delta_ratio, "reasons": reasons}

def clean_lfp(
    raw_data: mne.io.BaseRaw | str | Path, *, file_format: str | None = None, subject_id: str | None = None,
    config: CleaningConfig | None = None, return_removed: bool = False, return_raw: bool = False,
    spike_removal: bool = False, detect_spikes: bool = True, detect_dead_segments: bool = True,
    reconstruct_dead: bool | None = None,
) -> LFPCleanResult:
    cfg = config or CleaningConfig()
    if reconstruct_dead is None: reconstruct_dead = cfg.reconstruct_dead

    if isinstance(raw_data, (str, Path)):
        raw = load_raw(raw_data, file_format=file_format, preload=True, verbose="WARNING")
        path = Path(raw_data)
        if subject_id is None: subject_id = path.stem.split("_")[0] if path.stem else "unknown"
    else:
        # FIX: Prevent in-place mutation of the caller's object
        raw = raw_data.copy()
        if not raw.preload: raw.load_data()
        if subject_id is None: subject_id = "unknown"

    pick_idx = _resolve_picks(raw, cfg.picks)
    roi_names = [raw.ch_names[i] for i in pick_idx]
    sfreq = float(raw.info["sfreq"])

    if cfg.bandpass_l_freq is not None or cfg.bandpass_h_freq is not None:
        raw.filter(l_freq=cfg.bandpass_l_freq, h_freq=cfg.bandpass_h_freq, picks=pick_idx, verbose=False)

    eeg = raw.get_data(picks=pick_idx)
    eeg_original = eeg.copy()

    input_nan_mask = np.isnan(eeg)
    if input_nan_mask.any():
        if not reconstruct_dead:
            raise ValueError("Input contains NaN. Set reconstruct_dead=True to spectrally reconstruct gaps.")
        logger.info("Input NaN detected. Temporarily reconstructing for FFT notch...")
        for i in range(eeg.shape[0]):
            if input_nan_mask[i].any():
                eeg[i], _ = _reconstruct_dead_segments(eeg[i], input_nan_mask[i], sfreq, context_sec=cfg.recon_context_sec)

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
    ekg_peaks, ekg_templates, ekg_counts, ekg_thresholds, channel_qc = [], [], [], [], []

    for i, name in enumerate(roi_names):
        if detect_dead_segments:
            dead_masks[i] = _detect_dead_segments(notched[i], sfreq, window_sec=cfg.dead_segment_sec, std_ratio=cfg.dead_std_ratio, min_gap_merge_sec=cfg.min_gap_merge_sec, min_dead_duration_sec=cfg.min_dead_duration_sec)
            if reconstruct_dead and dead_masks[i].any():
                reconstructed_sig, recon_mask = _reconstruct_dead_segments(notched[i], dead_masks[i], sfreq, context_sec=cfg.recon_context_sec)
                recon_masks[i] = recon_mask
                input_for_ekg = reconstructed_sig
            else: input_for_ekg = notched[i].copy()
        else: input_for_ekg = notched[i].copy()

        cleaned_ekg, removed_ch, peaks_ch, template_ch, thr_used = _ekg_artifact_removal(
            input_for_ekg, fs=sfreq, thr=cfg.ekg_thr, polarity=cfg.ekg_polarity,
            auto_sensitivity=cfg.ekg_auto_sensitivity, subtract_t_wave=cfg.subtract_t_wave, qrs_highpass_hz=cfg.qrs_highpass_hz,
        )
        ekg_peaks.append(peaks_ch); ekg_templates.append(template_ch); ekg_counts.append(len(peaks_ch)); ekg_thresholds.append(thr_used)

        if detect_spikes:
            mask_ch = _detect_spikes(cleaned_ekg, sfreq, cfg.spike_segment_sec, cfg.spike_mad_thresh)
            spike_masks[i] = mask_ch
            cleaned[i] = _apply_spike_mask(cleaned_ekg, mask_ch) if spike_removal else cleaned_ekg
        else: cleaned[i] = cleaned_ekg

        if return_removed: removed[i] = removed_ch

        qc_metrics = _evaluate_channel_quality(notched[i], cleaned[i], dead_masks[i], spike_masks[i], sfreq, name, cfg.bandpass_l_freq)
        channel_qc.append(qc_metrics)

    if input_nan_mask.any():
        cleaned[input_nan_mask] = np.nan
        if return_removed: removed[input_nan_mask] = np.nan

    result: LFPCleanResult = {
        "data": cleaned[np.newaxis, :, :], "roi_names": roi_names, "subject_id": subject_id, "sfreq": sfreq,
        "ekg_peaks": ekg_peaks, "ekg_templates": ekg_templates, "ekg_counts": ekg_counts, "ekg_thresholds": ekg_thresholds,
        "channel_qc": channel_qc, "nan_mask": input_nan_mask[np.newaxis, :, :],
    }
    if return_raw: result["raw_filtered"] = eeg_original[np.newaxis, :, :]
    if return_removed: result["removed"] = removed[np.newaxis, :, :]
    if detect_spikes: result["spike_mask"] = spike_masks[np.newaxis, :, :]
    if detect_dead_segments: 
        result["dead_mask"] = dead_masks[np.newaxis, :, :]
        result["reconstructed_mask"] = recon_masks[np.newaxis, :, :]
    return result
