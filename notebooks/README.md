# Reference guide for the `lfp_toolkit` library. 

These tables detail every configuration parameter, execution flag, and core function, specifically highlighting the **Digital Signal Processing (DSP) rationale** and the **Sensor-Source Parity Rules** required for valid cross-modal comparisons.

<br>

### 1. `CleaningConfig` (Continuous Signal Processing)
This Pydantic model governs the transformation of raw continuous data. It is the most critical configuration for preventing "overprocessing" and ensuring mathematical parity with your EEG source pipeline.

| Parameter | Type | Default | DSP Rationale & Sensor-Source Parity Rule |
| :--- | :--- | :--- | :--- |
| **`picks`** | `str \| list[str]` | `"eeg"` | Defines which channel types to process. Use a list like `["seeg", "dbs", "ecog"]` for intracranial data to avoid processing irrelevant channels or triggering MNE type errors. |
| **`bandpass_l_freq`** | `float \| None` | `1.0` | **High-pass filter.** *Parity Rule:* Must exactly match your EEG source pipeline. Setting to `1.0` destroys slow cortical potentials like the Readiness Potential (<1 Hz). Use `0.1` or `None` if studying RP. |
| **`bandpass_h_freq`** | `float \| None` | `100.0` | **Low-pass filter.** *Parity Rule:* Must match EEG pipeline. Prevents aliasing and removes high-frequency noise/muscle artifacts. Set to `None` to disable. |
| **`notch_freq`** | `float \| None` | `60.0` | **Line-noise notch.** Applies a frequency-domain Gaussian soft-notch. Unlike IIR/FIR filters, this prevents time-domain ringing and phase distortion. Set to `None` to disable. |
| **`notch_max_db`** | `float` | `1.0` | Maximum attenuation (in dB) at the center of the notch frequency. A low value (like 1.0) creates a very "soft" attenuation that minimally disturbs neighboring neural frequencies. |
| **`notch_sigma_hz`** | `float` | `2.0` | Standard deviation (in Hz) of the Gaussian notch curve. Controls the width of the attenuation band around the line noise. |
| **`ekg_thr`** | `float \| "auto"` | `50e-6` | **EKG Peak Detection Threshold.** *Warning:* `"auto"` uses `Median + (2.0 * MAD)`. In task-based LFP, motor potentials will trigger false positives, causing the algorithm to subtract neural data. **Set to `1.0` (1 Volt)** to safely bypass EKG removal if EEG didn't use ICA. |
| **`ekg_auto_sensitivity`**| `float` | `2.0` | Multiplier for the MAD when `ekg_thr="auto"`. Lower values increase sensitivity to smaller EKG peaks but increase false positives. |
| **`ekg_polarity`** | `-1 \| 1` | `1` | Expected polarity of the QRS complex. Intracranial EKG artifacts can be inverted depending on the electrode montage and reference. |
| **`subtract_t_wave`** | `bool` | `False` | If `True`, models and subtracts the slower T-wave in addition to the sharp QRS complex. Usually unnecessary for LFP and risks overcleaning low-frequency neural data. |
| **`qrs_highpass_hz`** | `float` | `4.0` | High-pass cutoff used internally to isolate the sharp QRS spike from the broader QRST template during adaptive least-squares fitting. |
| **`dead_segment_sec`** | `float` | `0.2` | Window size (in seconds) for calculating local standard deviation to detect flatlines (electrode disconnects or saturations). |
| **`dead_std_ratio`** | `float` | `0.05` | Threshold ratio. If a window's std is less than 5% of the median signal std, it is flagged as a dead segment. |
| **`min_gap_merge_sec`** | `float` | `1.0` | If two dead segments are separated by less than this duration, they are merged into a single continuous dead segment. |
| **`min_dead_duration_sec`**| `float` | `0.3` | Minimum duration (in seconds) a flatline must last to be officially classified as a "dead segment" (ignores micro-pauses). |
| **`reconstruct_dead`** | `bool` | `True` | **Spectral Synthesis.** If `True`, fills flatlines with phase-randomized surrogate data matching the local PSD. *Parity Rule:* Set to `False` to replace with `NaN` and reject the entire epoch later, ensuring strict mathematical parity with source models. |
| **`recon_context_sec`** | `float` | `2.0` | The amount of healthy data (in seconds) before and after a dead segment used to estimate the PSD for spectral reconstruction. |
| **`spike_segment_sec`** | `float` | `0.5` | Window size for calculating local MAD (Median Absolute Deviation) to detect electrode pops or high-frequency spikes. |
| **`spike_mad_thresh`** | `float` | `8.0` | **Spike Threshold.** Multiplier for the robust standard deviation (MAD * 1.4826). Values exceeding `±8.0` robust SDs are flagged. Lowering to `5.0` is more aggressive but risks removing neural bursts. |

<br>

### 2. `EpochingConfig` (Trial Extraction & Edge Mitigation)
This model handles the transition from continuous time to discrete trials, specifically addressing the DSP pitfalls of wavelet edge effects and spectral leakage.

| Parameter | Type | Default | DSP Rationale & Sensor-Source Parity Rule |
| :--- | :--- | :--- | :--- |
| **`output_dir`** | `Path` | *Required* | Directory where the compressed `.npz` tensor will be saved. |
| **`lfp_file`** | `Path` | *Required* | Path to the raw continuous LFP file (EDF, FIF, BrainVision). |
| **`file_format`** | `str \| None` | `None` | Explicitly define format (`"edf"`, `"fif"`) if auto-detection via extension fails. |
| **`events_tsv`** | `Path` | *Required* | Path to the BIDS-compliant events file containing trial onsets and conditions. |
| **`alignment_column`** | `str` | *Required* | The column in the TSV (e.g., `"cue_onset"`) containing the exact timestamps (in seconds) to align time-zero. |
| **`subject_id`** | `str` | *Required* | Subject identifier used for file naming and metadata. |
| **`epoch_tmin`** | `float` | `-1.5` | Start time of the epoch (in seconds) relative to the alignment column. |
| **`epoch_tmax`** | `float` | `1.0` | End time of the epoch (in seconds) relative to the alignment column. |
| **`padding_sec`** | `float` | `0.5` | **Edge-Effect Mitigation.** Extracts extra data beyond `tmin`/`tmax` (e.g., extracts `[-2.0, 1.5]`). This provides the necessary context for Morlet wavelets to compute accurate low-frequency power at the true boundaries. |
| **`fade_duration_sec`** | `float` | `0.2` | **Spectral Leakage Prevention.** Applies a Hann window taper to the first and last $N$ milliseconds of the epoch, smoothly forcing the signal to zero at the boundaries before FFT operations. |
| **`target_sfreq`** | `float \| None` | `None` | **Resampling.** Downsamples or upsamples the epoch data. *Parity Rule:* Must exactly match the sampling rate of your EEG source tensor (e.g., `250.0` Hz). |
| **`cleaning_config`** | `CleaningConfig`| `None` | Injects the explicit cleaning rules into the epoching pipeline, ensuring the continuous data is cleaned identically to the QC step. |

<br>

### 3. `TFRConfig` (Time-Frequency Analysis)
This model governs the Morlet wavelet convolution, baseline normalization, and final visualization cropping.

| Parameter | Type | Default | DSP Rationale & Sensor-Source Parity Rule |
| :--- | :--- | :--- | :--- |
| **`tensor_file`** | `Path` | *Required* | Path to the `.npz` tensor generated by `convert_lfp_to_tensor`. |
| **`output_base`** | `Path` | *Required* | Base directory for saving TFR plots (`.png`) and HDF5 data files (`.h5`). |
| **`subject_id`** | `str` | *Required* | Subject identifier for directory structuring. |
| **`condition`** | `str` | *Required* | Condition label (e.g., `"gain"`, `"loss"`) used for sub-folder naming. |
| **`epoch_tmin` / `tmax`**| `float` | `-1.5` / `1.0`| The original boundaries of the tensor. Used internally to instantiate MNE `EpochsArray` with the correct time axis. |
| **`freqs`** | `list[float]` | `4` to `90` | Array of frequencies (in Hz) to evaluate using Morlet wavelets. |
| **`n_cycles_divisor`** | `float` | `3.0` | Calculates wavelet cycles as `freqs / divisor`. A divisor of `3.0` provides the optimal trade-off: good temporal resolution for high frequencies (Gamma) and good frequency resolution for low frequencies (Theta). |
| **`baseline_window`** | `tuple[float, float]`| `(-1.5, -1.0)`| **Percent-Change Normalization.** The time window used to compute the mean baseline power. *Parity Rule:* **Must exactly match** the baseline window used in your EEG LCMV pipeline, otherwise amplitude comparisons are mathematically invalid. |
| **`plot_tmin` / `tmax`** | `float` | `-1.0` / `1.0`| **Post-Convolution Cropping.** Crops the TFR *after* wavelet convolution. This ensures the final visualized data is free from the edge artifacts protected by `padding_sec`. |
| **`plot_fmin` / `fmax`** | `float` | `4.0` / `50.0` | Frequency bounds for the Y-axis of the final TFR plots. |
| **`vlim`** | `tuple[float, float]`| `(-1.0, 1.0)` | Symmetrical colorbar limits (in percent change) for the heatmap. Ensures consistent visual scaling across subjects and conditions. |
| **`anatomical_labels`** | `dict[str, str]` | `{}` | Maps raw channel names (e.g., `"R1"`) to anatomical labels (e.g., `"Right Ventral Striatum"`) for publication-ready plot titles. |
| **`show_plots`** | `bool` | `False` | If `True`, opens interactive matplotlib windows. Keep `False` for headless server/batch processing. |
| **`save_plots`** | `bool` | `True` | Saves the generated TFR heatmaps to the `output_base` directory. |

<br>

### 4. Core API Functions & Execution Flags

These are the primary functions exposed by the library, along with the critical boolean flags that alter their DSP behavior.

| Function / Flag | Purpose & DSP Implications |
| :--- | :--- |
| **`load_raw()`** | Abstracts MNE-Python's file readers. Standardizes loading of EDF, EGI, FIF, and BrainVision formats, handling BrainVision's strict `.vhdr` header requirements automatically. |
| **`clean_lfp()`** | The computational engine. Applies filtering, dead-segment detection, EKG subtraction, and spike masking. |
| ↳ `return_raw` | `bool`. If `True`, returns the filtered-but-uncleaned data. **Required** for `viz_data` to plot "Before vs After" comparisons. |
| ↳ `return_removed` | `bool`. If `True`, returns the exact signal that was subtracted (e.g., the EKG template). **Crucial** for verifying that neural data wasn't overcleaned. |
| ↳ `spike_removal` | `bool`. If `True`, replaces detected spikes with `NaN`. If `False`, spikes are flagged in the mask but left in the data (useful for QC visualization). |
| **`viz_data()`** | Generates multi-panel diagnostic plots (raw vs. cleaned traces, PSDs, and EKG templates) to allow manual verification of the automated cleaning pipeline. |
| **`convert_lfp_to_tensor()`**| Slices continuous data into discrete trials, applies padding/fades, rejects `NaN` trials, and saves a 4D tensor `(Subjects, ROIs, Epochs, Time)`. |
| **`generate_tfr_plots()`** | Computes Total Power Morlet TFRs, applies baseline correction, crops to `plot_tmin/tmax`, and generates publication-ready visualizations. |
| **`compute_roi_tfr()`** | Lower-level function to compute the TFR for a single ROI. Uses `average=True` to compute total power (Evoked + Induced) rather than just phase-locked evoked power. |
