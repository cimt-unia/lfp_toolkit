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
) -> mne.io.BaseRaw:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    
    if file_format is None:
        if ext == ".edf": file_format = "edf"
        elif ext in [".mff", ".egi", ".raw"]: file_format = "egi"
        elif ext == ".fif": file_format = "fif"
        elif ext in [".vhdr", ".eeg", ".vmrk"]: file_format = "brainvision"
        else: raise ValueError(f"Unsupported file extension: {ext}. Specify file_format explicitly.")
            
    file_format = file_format.lower()
    logger.info("Loading %s file: %s", file_format.upper(), path.name)
    
    if file_format == "edf":
        return mne.io.read_raw_edf(str(path), preload=preload, verbose=verbose)
    elif file_format == "egi":
        return mne.io.read_raw_egi(str(path), preload=preload, verbose=verbose)
    elif file_format == "fif":
        return mne.io.read_raw_fif(str(path), preload=preload, verbose=verbose)
    elif file_format == "brainvision":
        # FIX: BrainVision strictly requires the .vhdr header file.
        if ext in [".eeg", ".vmrk"]:
            vhdr_path = path.with_suffix(".vhdr")
            if not vhdr_path.exists():
                raise FileNotFoundError(f"BrainVision requires the .vhdr header file. Missing: {vhdr_path}")
            path = vhdr_path
        return mne.io.read_raw_brainvision(str(path), preload=preload, verbose=verbose)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
