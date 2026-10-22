# LFP ToolKit tutorial


The `lfp_toolkit` is designed to enforce strict Digital Signal Processing (DSP) standards for intracranial data.


<br>

### 1. Configuration Deep Dive

The library relies on three Pydantic models. Understanding the DSP rationale behind their parameters is critical for preventing overprocessing.

#### A. `CleaningConfig` (Signal Preservation)
This config governs the continuous data transformation. 
*   **`bandpass_l_freq` / `bandpass_h_freq`**: Defines the passband. *Rule:* Must exactly match the high-pass/low-pass filters applied to your EEG source data. A 1.0 Hz high-pass will destroy the Readiness Potential (RP).
*   **`notch_freq`**: Applies a frequency-domain Gaussian soft-notch. Unlike IIR/FIR notch filters, this prevents time-domain ringing and phase distortion.
*   **`ekg_thr`**: The threshold for QRS peak detection. *Warning:* Using `"auto"` calculates `Median + (2.0 * MAD)`. In task-based LFP, large motor potentials or beta bursts will exceed this threshold, causing the algorithm to build a "neural template" and subtract your cognitive signals. **Set to `1.0` (1 Volt) to bypass EKG removal** if your EEG pipeline did not explicitly remove EKG via ICA.
*   **`reconstruct_dead`**: If `True`, uses spectral synthesis (FFT/IFFT) to fill flatlines. If `False`, replaces dead segments with `NaN`. *Rule:* Set to `False` for strict sensor-source parity; let the epoching step reject `NaN` trials entirely.

#### B. `EpochingConfig` (Edge-Effect Mitigation)
This config handles the transition from continuous time to discrete trials.
*   **`padding_sec`**: Morlet wavelets require data beyond the epoch boundaries to compute accurate low-frequency power. Setting `padding_sec=0.5` extracts a larger window (e.g., `[-2.0, 1.5]`), computes the TFR, and then crops it to `[-1.5, 1.0]`. This prevents artificial power drops at the edges of your plots.
*   **`fade_duration_sec`**: Applies a Hann window taper to the first and last $N$ milliseconds of the epoch. This forces the signal to zero at the boundaries, preventing high-frequency spectral leakage during FFT operations.
*   **`target_sfreq`**: Resamples the data. Ensure this matches your EEG source tensor (e.g., `250.0` Hz).

#### C. `TFRConfig` (Time-Frequency Resolution)
*   **`freqs` & `n_cycles_divisor`**: Defines the wavelet parameters. `n_cycles = freqs / 3.0` is the standard trade-off, providing good temporal resolution for high frequencies (Gamma) and good frequency resolution for low frequencies (Theta/Alpha).
*   **`baseline_window`**: The time window used to compute the mean power for percent-change normalization. *Rule:* Must exactly match the baseline window used in your EEG LCMV pipeline.
*   **`plot_tmin` / `plot_tmax`**: Crops the final TFR *after* wavelet convolution. This ensures the visualized data is free from the edge artifacts protected by `padding_sec`.

<br>

### 2. The Refactored Production Pipeline

This script eliminates redundant processing, explicitly defines all DSP parameters to ensure parity with your EEG pipeline, and includes robust error handling.

```python
import logging
from pathlib import Path
from lfp_toolkit import (
    CleaningConfig, EpochingConfig, TFRConfig,
    convert_lfp_to_tensor, generate_tfr_plots, 
    load_raw, clean_lfp, viz_data
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] %(levelname)s: %(message)s", 
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main() -> None:
    # Define base directories
    data_dir = Path("./data")
    qc_dir = data_dir / "viz_qc"
    tensor_dir = data_dir / "tensors"
    tfr_dir = data_dir / "tfr_lfp"
    
    for d in [qc_dir, tensor_dir, tfr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. EXPLICIT CLEANING CONFIGURATION
    # ========================================================================
    # Crucial: These parameters MUST match your EEG source pipeline exactly.
    clean_cfg = CleaningConfig(
        picks=["seeg", "dbs", "ecog", "eeg", "misc"],
        bandpass_l_freq=1.0,       # Match EEG high-pass (Note: Destroys <1Hz RP)
        bandpass_h_freq=100.0,     # Match EEG low-pass
        notch_freq=60.0,           # Match EEG notch
        ekg_thr=1.0,               # 1.0 Volt bypasses EKG removal to prevent overcleaning
        subtract_t_wave=False,
        reconstruct_dead=False,    # Keep as NaN for strict epoch rejection
        spike_mad_thresh=8.0       # Conservative threshold to preserve neural bursts
    )

    # ========================================================================
    # 2. OPTIONAL: PRE-FLIGHT QC VISUALIZATION
    # ========================================================================
    # We run cleaning once manually to generate diagnostic plots before the 
    # computationally expensive epoching step.
    lfp_file = Path("/mnt/movement/users/jaizor/xtra/data/eeg/OCD_task/Dp_02/DP02_LFP_run1.edf")
    
    logger.info("Running pre-flight QC cleaning...")
    raw = load_raw(lfp_file, file_format="edf")
    
    # return_raw=True and return_removed=True are required for viz_data to 
    # plot the "Before vs After" and "Subtracted Template" panels.
    qc_result = clean_lfp(
        raw, 
        subject_id="DP02", 
        config=clean_cfg, 
        return_raw=True, 
        return_removed=True,
        detect_spikes=True,
        spike_removal=False # Keep spikes in data for QC visual comparison
    )
    
    viz_data(qc_result, save_dir=qc_dir, show=False)
    logger.info("QC plots saved to %s. Verify EKG templates and spike masks before proceeding.", qc_dir)

    # ========================================================================
    # 3. EPOCHING CONFIGURATION & TENSOR GENERATION
    # ========================================================================
    epoch_cfg = EpochingConfig(
        output_dir=tensor_dir,
        lfp_file=lfp_file,
        file_format="edf",
        events_tsv=Path("/mnt/movement/users/jaizor/xtra/data/eeg/OCD_task/Dp_02/DP02_EEG_run1_cue_events.tsv"),
        alignment_column="cue_onset",
        subject_id="DP02",
        epoch_tmin=-1.5,
        epoch_tmax=1.0,
        padding_sec=0.5,         # Extracts [-2.0, 1.5] to protect Morlet wavelet edges
        fade_duration_sec=0.1,   # 100ms Hann taper to prevent spectral leakage
        target_sfreq=250.0,      # Match EEG source tensor sampling rate
        cleaning_config=clean_cfg # Injects the exact same cleaning rules used in QC
    )

    logger.info("Starting one-shot cleaning and epoching pipeline...")
    # This function loads the raw EDF, applies clean_cfg, extracts epochs, 
    # applies fades, rejects NaN/artifact trials, and saves the 4D tensor.
    tensor_path = convert_lfp_to_tensor(epoch_cfg)

    # ========================================================================
    # 4. TFR CONFIGURATION & PLOTTING
    # ========================================================================
    tfr_cfg = TFRConfig(
        tensor_file=tensor_path,
        output_base=tfr_dir,
        subject_id=epoch_cfg.subject_id,
        condition="Choice",
        epoch_tmin=epoch_cfg.epoch_tmin,
        epoch_tmax=epoch_cfg.epoch_tmax,
        baseline_window=(-1.5, -1.0), # MUST match EEG source baseline window
        plot_tmin=-1.0,               # Crop to [-1.0, 1.0] AFTER wavelet convolution
        plot_tmax=1.0,
        anatomical_labels={
            "R1": "Right Ventral Striatum", 
            "R2": "Right Orbitofrontal Cortex",
            "L1": "Left Ventral Striatum / ALIC", 
            "L2": "Left Ventral Striatum / ALIC",
        },
    )

    logger.info("Computing Time-Frequency Representations...")
    generate_tfr_plots(tfr_cfg)
    
    logger.info("Pipeline complete. Tensors at %s | TFRs at %s", tensor_dir, tfr_dir)

if __name__ == "__main__":
    main()
```

<br>

### 3. Critical DSP Rules for Sensor-Source Parity

When comparing the output of this script against your EEG LCMV beamformer tensors, you must enforce the following invariants:

1.  **The Baseline Invariant**: The `baseline_window` in `TFRConfig` dictates the divisor for percent-change normalization. If your EEG pipeline baselines at `(-1.5, -1.0)` and your LFP pipeline baselines at `(-1.0, 0.0)`, a 20% power increase in the LFP plot is mathematically incomparable to a 20% increase in the EEG plot.
2.  **The Trial Intersection**: `convert_lfp_to_tensor` will automatically reject any epoch containing a `NaN` (from spike removal) or a dead segment. You must ensure that your EEG pipeline rejects the **exact same trial indices**. If the EEG pipeline keeps 50 trials and the LFP pipeline keeps 54, the Signal-to-Noise Ratio (SNR) will differ, invalidating direct amplitude comparisons.
3.  **The Post-Convolution Crop**: Notice that `epoch_tmin` is `-1.5` but `plot_tmin` is `-1.0`. The library computes the wavelet convolution on the full `[-1.5, 1.0]` window (plus the `0.5s` padding). It then crops the result to `[-1.0, 1.0]` for visualization. This guarantees that the edges of your published plots are mathematically pure and free from wavelet edge-effect artifacts.
