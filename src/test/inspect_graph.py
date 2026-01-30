#!/usr/bin/env python3
"""
Inspect generated graphs and clusters.
Visualize structure and relationships.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"


def load_graph(db_id: str) -> nx.Graph | None:
    """Load graph for a database."""
    graph_path = PROCESSED_DIR / "db_graphs" / f"{db_id}.pkl"
    if not graph_path.exists():
        return None
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def load_clusters(db_id: str) -> dict | None:
    """Load clusters for a database."""
    cluster_path = PROCESSED_DIR / "clusters" / f"{db_id}.json"
    if not cluster_path.exists():
        return None
    with open(cluster_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_graph_info(G: nx.Graph, db_id: str) -> None:
    """Print graph structure information."""
    print("=" * 60)
    print(f"Graph Structure: {db_id}")
    print("=" * 60)
    
    print(f"\nNodes ({G.number_of_nodes()}):")
    for node in sorted(G.nodes()):
        print(f"  - {node}")
    
    print(f"\nEdges ({G.number_of_edges()}):")
    hard_edges = []
    soft_edges = []
    
    for u, v, data in G.edges(data=True):
        edge_type = data.get("type", "unknown")
        weight = data.get("weight", 0.0)
        edge_str = f"  {u} <--[{edge_type}, w={weight:.3f}]--> {v}"
        
        if edge_type == "hard":
            hard_edges.append(edge_str)
        else:
            soft_edges.append(edge_str)
    
    if hard_edges:
        print("\n  Hard Edges (FK):")
        for edge in hard_edges:
            print(edge)
    
    if soft_edges:
        print("\n  Soft Edges (Semantic):")
        for edge in soft_edges:
            print(edge)
    
    # Graph statistics
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Hard edges: {len(hard_edges)}")
    print(f"  Soft edges: {len(soft_edges)}")
    print(f"  Density: {nx.density(G):.3f}")
    
    # Connected components
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")
    if len(components) > 1:
        print("  (Graph is disconnected)")
        for i, comp in enumerate(components):
            print(f"    Component {i}: {sorted(comp)}")


def print_cluster_info(clusters: dict, db_id: str) -> None:
    """Print cluster information."""
    print("\n" + "=" * 60)
    print(f"Cluster Information: {db_id}")
    print("=" * 60)
    
    partition = clusters.get("partition", {})
    cluster_vectors = clusters.get("cluster_vectors", {})
    num_clusters = clusters.get("num_clusters", 0)
    
    print(f"\nNumber of clusters: {num_clusters}")
    
    # Group tables by cluster
    cluster_groups = {}
    for table, c_id in partition.items():
        if c_id not in cluster_groups:
            cluster_groups[c_id] = []
        cluster_groups[c_id].append(table)
    
    print(f"\nCluster Details:")
    for c_id in sorted(cluster_groups.keys()):
        members = cluster_groups[c_id]
        vector_dim = len(cluster_vectors.get(str(c_id), []))
        print(f"\n  Cluster {c_id} ({len(members)} tables):")
        print(f"    Tables: {', '.join(sorted(members))}")
        print(f"    Centroid vector dimension: {vector_dim}")


def list_available_dbs() -> list[str]:
    """List all available database IDs."""
    graph_dir = PROCESSED_DIR / "db_graphs"
    if not graph_dir.exists():
        return []
    
    db_ids = []
    for pkl_file in graph_dir.glob("*.pkl"):
        db_id = pkl_file.stem
        if (PROCESSED_DIR / "clusters" / f"{db_id}.json").exists():
            db_ids.append(db_id)
    
    return sorted(db_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated graphs and clusters")
    parser.add_argument(
        "--db-id",
        type=str,
        help="Database ID to inspect (if not provided, lists all available)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available database IDs",
    )
    args = parser.parse_args()
    
    if args.list or not args.db_id:
        print("Available databases:")
        db_ids = list_available_dbs()
        if db_ids:
            for db_id in db_ids:
                print(f"  - {db_id}")
        else:
            print("  No databases found. Run 'python -m src.offline.build_graph' first.")
        return
    
    db_id = args.db_id
    
    # Load graph
    G = load_graph(db_id)
    if not G:
        print(f"Error: Graph not found for {db_id}")
        print("Run 'python -m src.test.test_bird_mini' first.")
        return
    
    # Load clusters
    clusters = load_clusters(db_id)
    if not clusters:
        print(f"Error: Clusters not found for {db_id}")
        return
    
    # Print information
    print_graph_info(G, db_id)
    print_cluster_info(clusters, db_id)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
