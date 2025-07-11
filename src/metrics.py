import numpy as np
import networkx as nx

from src.connectivity import prepare_connectivity_matrix


def compute_degree_strength(con, n_channels):
    """Compute degree strength for each node in the connectivity network"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    degree_strength = np.array([G.degree(node, weight='weight') for node in G.nodes()])
    
    return degree_strength


def compute_betweenness_centrality(con, n_channels):
    """Compute betweenness centrality for each node in the connectivity network"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    return np.array(list(betweenness.values()))


def compute_clustering_coefficient(con, n_channels):
    """Compute clustering coefficient for each node in the connectivity network"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    clustering = nx.clustering(G, weight='weight')
    
    return np.array(list(clustering.values()))


def compute_eigenvector_centrality(con, n_channels):
    """Compute eigenvector centrality for each node in the connectivity network"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    eigenvector = nx.eigenvector_centrality(G, weight='weight')
    
    return np.array(list(eigenvector.values()))


def analyze_hemisphere_epileptogenic_zone(con, n_channels, ch_names, metric_func=None):
    """Analyze if the epileptogenic zone is in the Right or Left Hemisphere"""
    if metric_func is None:
        metric_func = compute_degree_strength
    
    # Compute the specified metric
    metric_values = metric_func(con, n_channels)
    
    # Convert to numpy array if needed
    metric_values = np.array(metric_values)
    
    # Define hemisphere mapping based on 10-20 system
    left_hemisphere = ['C3', 'F3', 'F7', 'Fp1', 'P3', 'T3', 'T5']
    right_hemisphere = ['C4', 'F4', 'F8', 'Fp2', 'P4', 'T4', 'T6']
    midline = ['Cz', 'Fpz', 'Pz', 'O1', 'O2']  # O1/O2 are bilateral occipital
    
    # Group channels by hemisphere
    left_indices = [i for i, ch in enumerate(ch_names) if ch in left_hemisphere]
    right_indices = [i for i, ch in enumerate(ch_names) if ch in right_hemisphere]
    midline_indices = [i for i, ch in enumerate(ch_names) if ch in midline]
    
    # Calculate hemisphere metrics
    left_values = metric_values[left_indices] if left_indices else np.array([])
    right_values = metric_values[right_indices] if right_indices else np.array([])
    midline_values = metric_values[midline_indices] if midline_indices else np.array([])
    
    # Analyze hemisphere dominance
    left_mean = np.mean(left_values) if len(left_values) > 0 else 0
    right_mean = np.mean(right_values) if len(right_values) > 0 else 0
    midline_mean = np.mean(midline_values) if len(midline_values) > 0 else 0
    
    # Determine dominant hemisphere
    if left_mean > right_mean:
        dominant_hemisphere = "Left"
        dominance_ratio = left_mean / right_mean if right_mean > 0 else np.inf
    elif right_mean > left_mean:
        dominant_hemisphere = "Right"
        dominance_ratio = right_mean / left_mean if left_mean > 0 else np.inf
    else:
        dominant_hemisphere = "Bilateral"
        dominance_ratio = 1.0
    
    # Find most active channels
    max_idx = np.argmax(metric_values)
    most_active_channel = ch_names[max_idx]
    
    # Determine hemisphere of most active channel
    if most_active_channel in left_hemisphere:
        most_active_hemisphere = "Left"
    elif most_active_channel in right_hemisphere:
        most_active_hemisphere = "Right"
    else:
        most_active_hemisphere = "Midline"
    
    return {
        'dominant_hemisphere': dominant_hemisphere,
        'dominance_ratio': dominance_ratio,
        'left_mean': left_mean,
        'right_mean': right_mean,
        'midline_mean': midline_mean,
        'most_active_channel': most_active_channel,
        'most_active_hemisphere': most_active_hemisphere,
        'left_channels': [ch_names[i] for i in left_indices],
        'right_channels': [ch_names[i] for i in right_indices],
        'midline_channels': [ch_names[i] for i in midline_indices],
        'metric_values': metric_values
    }


def analyze_multiband_hemisphere_epileptogenic_zone(con_results, metric_func=None):
    """Analyze hemisphere epileptogenic zone across multiple frequency bands"""
    if metric_func is None:
        metric_func = compute_degree_strength
    
    # Get number of channels from first band
    first_band = next(iter(con_results.values()))
    n_channels = len(first_band["ch_names"])
    ch_names = first_band["ch_names"]
    
    # Analyze each frequency band
    results = {}
    for band_name, result in con_results.items():
        connectivity = result["connectivity"].squeeze()
        
        # Analyze hemisphere for this band
        band_analysis = analyze_hemisphere_epileptogenic_zone(
            connectivity, n_channels, ch_names, metric_func
        )
        
        results[band_name] = band_analysis
    
    # Summary analysis across all bands
    band_dominance = {}
    for band_name, analysis in results.items():
        band_dominance[band_name] = analysis['dominant_hemisphere']
    
    # Count hemisphere dominance across bands
    left_count = sum(1 for dom in band_dominance.values() if dom == 'Left')
    right_count = sum(1 for dom in band_dominance.values() if dom == 'Right')
    bilateral_count = sum(1 for dom in band_dominance.values() if dom == 'Bilateral')
    
    # Overall hemisphere conclusion
    if left_count > right_count:
        overall_dominance = "Left"
    elif right_count > left_count:
        overall_dominance = "Right"
    else:
        overall_dominance = "Mixed"
    
    # Find most consistent channels across bands
    all_most_active = [analysis['most_active_channel'] for analysis in results.values()]
    channel_counts = {}
    for ch in all_most_active:
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    
    most_frequent_channel = max(channel_counts, key=channel_counts.get)
    
    # Calculate average dominance ratios
    avg_dominance_ratios = {}
    for band_name, analysis in results.items():
        ratio = analysis['dominance_ratio']
        if ratio != np.inf:
            avg_dominance_ratios[band_name] = ratio
    
    overall_avg_ratio = np.mean(list(avg_dominance_ratios.values())) if avg_dominance_ratios else 1.0
    
    return {
        'band_results': results,
        'overall_dominance': overall_dominance,
        'band_dominance_summary': {
            'left_count': left_count,
            'right_count': right_count,
            'bilateral_count': bilateral_count,
            'total_bands': len(con_results)
        },
        'most_frequent_active_channel': most_frequent_channel,
        'channel_frequency': channel_counts,
        'average_dominance_ratio': overall_avg_ratio,
        'band_specific_ratios': avg_dominance_ratios
    }
