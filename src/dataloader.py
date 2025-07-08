import mne
import pandas as pd

from pathlib import Path


mne.set_log_level("WARNING")


def load_csv(filepath, scaling_factor=1e-3):
    """Load eegdata data from a csv file."""
    path = Path(filepath)
    data = pd.read_csv(path, header=None).to_numpy().T
    data = data * scaling_factor
    return data


def create_raw_object(data, sfreq, ch_names, date):
    """Create mne.io.Raw object from numpy array"""
    info = mne.create_info(list(ch_names), sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(date)
    raw.set_montage('standard_1020')
    return raw


def load_raw(filepath):
    """Load mne.io.RawArray"""
    return mne.io.read_raw_fif(filepath, preload=True)


def save_raw(raw, filepath):
    """Save mne.io.RawArray"""
    raw.save(filepath, overwrite=True)
