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


def compute_global_efficiency(con, n_channels):
    """Compute global efficiency of the connectivity network"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    global_efficiency = nx.global_efficiency(G)
    
    return global_efficiency


def compute_modularity(con, n_channels):
    """Compute modularity of the connectivity network using Louvain algorithm"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    communities = nx.community.louvain_communities(G, weight='weight')
    modularity = nx.community.modularity(G, communities, weight='weight')
    
    return modularity


def compute_eigenvector_centrality(con, n_channels):
    """Compute eigenvector centrality for each node in the connectivity network"""
    conn_matrix = prepare_connectivity_matrix(con, n_channels)
    
    G = nx.from_numpy_array(conn_matrix)
    
    eigenvector = nx.eigenvector_centrality(G, weight='weight')
    
    return np.array(list(eigenvector.values()))
