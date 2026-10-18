# LFP toolkit

This package provides functions for cleaning signals, extracting epochs, and analyzing time-frequency representations for Local Field Potential data.

<br>

## Installation

Install the latest version directly from the repository:

```bash
python -m pip install --user git+https://github.com/cimt-unia/lfp_toolkit.git
```

*Note: The `git+` prefix is required by pip to resolve and clone version control URLs.*

For local development:

```bash
git clone https://github.com/cimt-unia/lfp_toolkit.git
cd lfp_toolkit
python -m pip install -e .
```

<br>

## Usage

The pipeline consists of loading raw data, applying signal cleaning, extracting epochs into a 4D tensor, and computing time-frequency representations.

```python
from pathlib import Path
from lfp_toolkit import (
    CleaningConfig, EpochingConfig, TFRConfig,
    load_raw, clean_lfp, viz_data, 
    convert_lfp_to_tensor, generate_tfr_plots
)

# Define configurations
clean_cfg = CleaningConfig(
    bandpass_l_freq=1.0, 
    bandpass_h_freq=100.0,
    notch_freq=60.0, 
    reconstruct_dead=False
)

epoch_cfg = EpochingConfig(
    output_dir=Path("./data"),
    lfp_file=Path("/path/to/subject_LFP.edf"),
    file_format="edf",
    events_tsv=Path("/path/to/events.tsv"),
    alignment_column="cue_onset",
    subject_id="SUB01",
    epoch_tmin=-1.5, 
    epoch_tmax=1.0
)

tfr_cfg = TFRConfig(
    tensor_file=Path("./data/tensor_SUB01_lfp_epochs.npz"),
    output_base=Path("./data/tfr_lfp"),
    subject_id="SUB01", 
    condition="Choice",
    anatomical_labels={"R1": "Right Ventral Striatum", "L1": "Left ALIC"}
)

# Execute pipeline
raw = load_raw(epoch_cfg.lfp_file, file_format=epoch_cfg.file_format)
clean_result = clean_lfp(raw, config=clean_cfg, return_raw=True)

viz_data(clean_result, save_dir=Path("./data/qc_plots"), show=False)

tensor_path = convert_lfp_to_tensor(epoch_cfg)

generate_tfr_plots(tfr_cfg)
```

<br>

## Architecture

The library is decoupled into single-responsibility modules.

| Module | Description |
|---|---|
| `io` | Unified wrapper around `mne.io` for EDF, FIF, EGI, and BrainVision formats. |
| `cleaning` | Signal processing engine. Handles filtering, dead-segment reconstruction, and EKG subtraction. |
| `viz` | Matplotlib-based quality control tools for time-series, PSDs, and templates. |
| `epoching` | Trial extraction, dead-segment rejection, and tensor serialization. |
| `tfr` | Morlet wavelet convolution and TFR visualization. |
| `config` | Pydantic models for strict parameter validation and serialization. |

<br>

## Configuration

Parameters are managed via Pydantic `BaseModel` classes (`CleaningConfig`, `EpochingConfig`, `TFRConfig`). This enforces type checking at runtime and prevents invalid pipeline states.

```python
from lfp_toolkit import CleaningConfig

cfg = CleaningConfig(
    ekg_thr="auto",
    subtract_t_wave=False
)
```

<br>

## Dependencies

- `numpy`
- `scipy`
- `mne`
- `pandas`
- `matplotlib`
- `pydantic`
