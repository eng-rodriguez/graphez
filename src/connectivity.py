import numpy as np

from mne_connectivity import spectral_connectivity_epochs

def functional_connectivity_analysis(epochs, method="coh"):
    """Perform functional connectivity analysis on mne.Epochs"""
    sfreq = epochs.info["sfreq"]
    freq_bands = {'delta': (1, 4),'theta': (4, 8),'alpha': (8, 13),'beta': (13, 30),'gamma': (30, 100)}
    
    # Compute connectivity
    connectivity_results = {}
    for band_name, (fmin_band, fmax_band) in freq_bands.items():
        con = spectral_connectivity_epochs(epochs, method=method, sfreq=sfreq, fmin=fmin_band, fmax=fmax_band, faverage=True, mt_adaptive=True)
        
        connectivity_results[band_name] = {
            "connectivity": con.get_data(),
            "freqs": con.freqs,
            "method": method,
            "n_epochs": con.n_epochs_used
        }
    
    return connectivity_results


def prepare_connectivity_matrix(con_matrix, n_channels):
    """Prepare connectivity data by ensuring proper 2D shape"""
    con_matrix = con_matrix.squeeze()
    if con_matrix.ndim == 1:
        # MNE-connectivity returns lower-triangular connectivity values
        # We need to reconstruct the full symmetric matrix
        full_matrix = np.zeros((n_channels, n_channels))
        
        # Fill lower triangle (excluding diagonal)
        idx = 0
        for i in range(n_channels):
            for j in range(i):
                full_matrix[i, j] = con_matrix[idx]
                full_matrix[j, i] = con_matrix[idx]  # Make symmetric
                idx += 1
        
        return full_matrix
    
    return con_matrix


def calculate_connectivity_threshold(filtered_con, threshold, valid_channels):
    """Calculate proportional connectivity threshold to keep top X% of connections"""
    connection_strengths = []
    for i in range(len(valid_channels)):
        for j in range(i + 1, len(valid_channels)):
            # Get all connection strengths from upper triangle
            connection_strengths.append(abs(filtered_con[i,j]))
    
    if connection_strengths:
        connection_strengths.sort(reverse=True)
        cutoff_index = min(int(threshold * len(connection_strengths)), len(connection_strengths) - 1)
        return connection_strengths[cutoff_index]
    else:
        return 0
