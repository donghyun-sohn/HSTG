#!/usr/bin/env python3
"""
Community detection and super node creation using Louvain algorithm.
Generates cluster centroids for efficient retrieval.
"""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np

try:
    import community.community_louvain as community_louvain
except ImportError:
    print("Warning: python-louvain not installed. Install with: pip install python-louvain")
    community_louvain = None


def detect_communities(G: nx.Graph) -> Tuple[Dict[str, int], Dict[int, np.ndarray]]:
    """
    Detect communities using Louvain algorithm and compute cluster centroids.
    
    Args:
        G: NetworkX graph with nodes containing 'vector' attribute
        
    Returns:
        Tuple of (partition, cluster_vectors)
        - partition: Dict mapping table_name -> cluster_id
        - cluster_vectors: Dict mapping cluster_id -> centroid_vector
    """
    if community_louvain is None:
        # Fallback: assign each node to its own cluster
        partition = {node: i for i, node in enumerate(G.nodes())}
        cluster_vectors = {
            i: G.nodes[node]["vector"] for i, node in enumerate(G.nodes())
        }
        return partition, cluster_vectors
    
    # 1. Run Louvain algorithm (uses 'weight' attribute)
    partition = community_louvain.best_partition(G, weight="weight")
    
    # 2. Compute cluster centroids
    cluster_vectors = {}
    cluster_groups: Dict[int, list] = {}  # {0: ['tab1', 'tab2'], 1: ['tab3']...}
    
    # Group tables by cluster
    for table, c_id in partition.items():
        if c_id not in cluster_groups:
            cluster_groups[c_id] = []
        cluster_groups[c_id].append(table)
        
    # Compute vector average (centroid) for each cluster
    for c_id, members in cluster_groups.items():
        vectors = [G.nodes[table]["vector"] for table in members]
        centroid = np.mean(vectors, axis=0)
        cluster_vectors[c_id] = centroid
        
    return partition, cluster_vectors
