
from mne.preprocessing import ICA

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
