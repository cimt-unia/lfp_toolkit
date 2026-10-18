# lfp_toolkit/io.py

from __future__ import annotations

import logging
from pathlib import Path
import mne

logger = logging.getLogger(__name__)

def load_raw(
    file_path: str | Path,
    file_format: str | None = None,
    preload: bool = True,
    verbose: str | bool = "WARNING",
) -> mne.io.Raw:
    """
    Load raw electrophysiological data from various file formats.
    
    Args:
        file_path: Path to the data file.
        file_format: Explicit format ('edf', 'egi', 'fif', 'brainvision'). 
                     If None, inferred from extension.
        preload: Whether to preload data into memory.
        verbose: MNE logging verbosity.
        
    Returns:
        mne.io.Raw object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    
    if file_format is None:
        if ext == ".edf":
            file_format = "edf"
        elif ext in [".mff", ".egi"]:
            file_format = "egi"
        elif ext == ".fif":
            file_format = "fif"
        elif ext in [".vhdr", ".eeg", ".vmrk"]:
            file_format = "brainvision"
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                "Please specify file_format explicitly."
            )
            
    logger.info("Loading %s file: %s", file_format.upper(), path.name)
    
    if file_format == "edf":
        return mne.io.read_raw_edf(str(path), preload=preload, verbose=verbose)
    elif file_format == "egi":
        return mne.io.read_raw_egi(str(path), preload=preload, verbose=verbose)
    elif file_format == "fif":
        return mne.io.read_raw_fif(str(path), preload=preload, verbose=verbose)
    elif file_format == "brainvision":
        return mne.io.read_raw_brainvision(str(path), preload=preload, verbose=verbose)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")