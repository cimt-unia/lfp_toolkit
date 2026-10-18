import logging
from pathlib import Path
from lfp_toolkit import (
    EpochingConfig, TFRConfig, convert_lfp_to_tensor, 
    generate_tfr_plots, load_raw, clean_lfp, viz_data
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def main() -> None:
    # 1. Define Configurations
    epoch_cfg = EpochingConfig(
        output_dir=Path("./data"),
        lfp_file=Path("/mnt/movement/users/jaizor/xtra/data/eeg/OCD_task/Dp_02/DP02_LFP_run1.edf"),
        file_format="edf",  # Explicitly specify or leave None for auto-detection
        events_tsv=Path("/mnt/movement/users/jaizor/xtra/data/eeg/OCD_task/Dp_02/DP02_EEG_run1_cue_events.tsv"),
        alignment_column="cue_onset",
        subject_id="DP02",
    )

    # 2. (Optional) Quick QC Visualization
    raw = load_raw(epoch_cfg.lfp_file, file_format=epoch_cfg.file_format)
    clean_result = clean_lfp(raw, subject_id="DP02", reconstruct_dead=False)
    viz_data(clean_result, save_dir=Path("./data/viz_qc"), show=False)

    # 3. Convert LFP to Tensors (Epochs)
    tensor_path = convert_lfp_to_tensor(epoch_cfg)

    # 4. Define TFR Configuration
    tfr_cfg = TFRConfig(
        tensor_file=tensor_path,
        output_base=Path("./data/tfr_lfp"),
        subject_id=epoch_cfg.subject_id,
        condition="Choice",
        epoch_tmin=epoch_cfg.epoch_tmin,
        epoch_tmax=epoch_cfg.epoch_tmax,
        anatomical_labels={
            "R1": "Right Ventral Striatum", "R2": "Right Orbitofrontal Cortex",
            "L1": "Left Ventral Striatum / ALIC", "L2": "Left Ventral Striatum / ALIC",
        },
    )

    # 5. Generate TFR Plots
    generate_tfr_plots(tfr_cfg)

if __name__ == "__main__":
    main()