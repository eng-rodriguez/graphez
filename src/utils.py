def drop_channels(raw, bad_channels):
    """Remove specified channels from mne.io.RawArray."""
    raw_clean = raw.copy()
    bads = [channel for channel in bad_channels if channel in raw_clean.ch_names]
    raw_clean.drop_channels(bads)
    return raw_clean