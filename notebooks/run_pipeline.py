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
