import mne
import numpy as np

from mne.preprocessing import ICA
from datetime import datetime

def apply_notch(raw, notch_freqs=[60.0, 120.0]):
    """Apply notch filter to remove powerline and harmonic noise"""
    raw_clean = raw.copy()
    raw_clean.notch_filter(freqs=notch_freqs)
    return raw_clean


def apply_bandpass(raw, l_freq=0.1, h_freq=40.0):
    """Apply bandpass filter to the mne.io.RawArray"""
    raw_clean = raw.copy()
    raw_clean.filter(l_freq=l_freq, h_freq=h_freq, method='iir')
    return raw_clean


def apply_reference(raw, reference):
    """Apply re-reference to the mne.io.RawArray"""
    raw_clean = raw.copy()
    raw_clean = raw_clean.set_eeg_reference(reference)
    return raw_clean


def fit_ica(raw, method="fastica", max_iter=5000):
    """Fit ICA decompisition in mne.io.RawArray"""
    ica = ICA(method=method, max_iter=max_iter, random_state=42)
    ica.fit(raw)
    return ica
    

def apply_ica(raw, ica, bad_components):
    """Reconstruct mne.io.RawArray with ICA decomposition"""
    ica.exclude = bad_components
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean


def create_epochs_object(raw, spike_time, epoch_length=3.0, baseline=None, preload=True):
    """Create three types of epochs around a spike event from mne.io.RawArray"""
    spike_sample = _parse_spike_time(spike_time, raw)
    
    sfreq = raw.info["sfreq"]
    epoch_samples = int(epoch_length * sfreq)
    
    events, event_id = _create_epoch_events(spike_sample, epoch_samples, raw.n_times)
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id, 0, epoch_length, baseline=baseline, preload=preload)
    
    # Split epochs by condition with consistent naming
    epochs_dict = {}
    if "before_spike" in event_id:
        epochs_dict["before_spike"] = epochs["before_spike"]
        
    if "during_spike" in event_id:
        epochs_dict["during_spike"] = epochs["during_spike"]
    
    if "after_spike" in event_id:
        epochs_dict["after_spike"] = epochs["after_spike"]
    
    return epochs_dict


def _parse_spike_time(spike_time, raw):
    """Convert spike time string to sample index"""
    spike_hour, spike_minute, spike_second = map(int, spike_time.split(":"))
    meas_date = raw.info["meas_date"]
    
    # Create spike datetime with same timezone as meas_date
    spike_datetime = datetime.combine(
        meas_date.date(),
        datetime.min.time().replace(hour=spike_hour, minute=spike_minute, second=spike_second)
    )
    
    # Make spike_datetime timezone-aware to match meas_date
    if meas_date.tzinfo is not None:
        spike_datetime = spike_datetime.replace(tzinfo=meas_date.tzinfo)
    
    time_diff = (spike_datetime - meas_date).total_seconds()
    
    sfreq = raw.info["sfreq"]
    spike_sample = int(time_diff * sfreq)
    
    return spike_sample


def _create_epoch_events(spike_sample, epoch_samples, raw_length):
    """Generate events array for before/during/after spike"""
    half_epoch_samples = epoch_samples // 2
    events_list = []
    event_id = {}
    
    # 1. Before spike: epoch ends at spike time
    before_start = spike_sample - epoch_samples
    if before_start >= 0:
        events_list.append([before_start, 0, 1])
        event_id["before_spike"] = 1
        
    # 2. During spike: epoch centered on spike time
    during_start = spike_sample - half_epoch_samples
    if during_start >= 0 and during_start + epoch_samples <= raw_length:
        events_list.append([during_start, 0, 2])
        event_id["during_spike"] = 2
    
    # 3. After spike: epoch starts at spike time
    after_start = spike_sample
    if after_start + epoch_samples <= raw_length:
        events_list.append([after_start, 0, 3])
        event_id["after_spike"] = 3
    
    return np.array(events_list), event_id
