#!/usr/bin/env python3
"""
Compare binary vs reinforced graph building strategies.
Shows differences in edge weights and clustering results.
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


def load_graph(db_id: str, strategy: str) -> nx.Graph | None:
    """Load graph for a database and strategy."""
    graph_path = PROCESSED_DIR / "db_graphs" / f"{db_id}_{strategy}.pkl"
    if not graph_path.exists():
        return None
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def load_clusters(db_id: str, strategy: str) -> dict | None:
    """Load clusters for a database and strategy."""
    cluster_path = PROCESSED_DIR / "clusters" / f"{db_id}_{strategy}.json"
    if not cluster_path.exists():
        return None
    with open(cluster_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_strategies(db_id: str) -> None:
    """Compare binary and reinforced strategies for a database."""
    print("=" * 70)
    print(f"Strategy Comparison: {db_id}")
    print("=" * 70)
    
    G_binary = load_graph(db_id, "binary")
    G_reinforced = load_graph(db_id, "reinforced")
    clusters_binary = load_clusters(db_id, "binary")
    clusters_reinforced = load_clusters(db_id, "reinforced")
    
    if not G_binary or not G_reinforced:
        print(f"Error: Graphs not found. Run build_graph with both strategies first.")
        return
    
    # Compare edges
    print("\n📊 Edge Comparison:")
    print("-" * 70)
    
    # Get all edges from both graphs
    binary_edges = {(u, v): d for u, v, d in G_binary.edges(data=True)}
    reinforced_edges = {(u, v): d for u, v, d in G_reinforced.edges(data=True)}
    
    all_edges = set(binary_edges.keys()) | set(reinforced_edges.keys())
    
    print(f"{'Edge':<40} {'Binary':<20} {'Reinforced':<20}")
    print("-" * 70)
    
    for u, v in sorted(all_edges):
        edge_str = f"{u} <-> {v}"
        b_data = binary_edges.get((u, v), {})
        r_data = reinforced_edges.get((u, v), {})
        
        b_weight = b_data.get("weight", 0.0)
        b_type = b_data.get("type", "none")
        r_weight = r_data.get("weight", 0.0)
        r_type = r_data.get("type", "none")
        
        b_str = f"{b_type} ({b_weight:.3f})"
        r_str = f"{r_type} ({r_weight:.3f})"
        
        diff = r_weight - b_weight
        diff_str = f"Δ{diff:+.3f}" if diff != 0 else ""
        
        print(f"{edge_str:<40} {b_str:<20} {r_str:<20} {diff_str}")
    
    # Compare clustering
    print("\n\n🔗 Clustering Comparison:")
    print("-" * 70)
    
    if clusters_binary and clusters_reinforced:
        partition_b = clusters_binary.get("partition", {})
        partition_r = clusters_reinforced.get("partition", {})
        
        num_clusters_b = clusters_binary.get("num_clusters", 0)
        num_clusters_r = clusters_reinforced.get("num_clusters", 0)
        
        print(f"Number of clusters:")
        print(f"  Binary:     {num_clusters_b}")
        print(f"  Reinforced: {num_clusters_r}")
        print(f"  Difference: {num_clusters_r - num_clusters_b:+d}")
        
        # Group tables by cluster
        clusters_b = {}
        clusters_r = {}
        
        for table, c_id in partition_b.items():
            if c_id not in clusters_b:
                clusters_b[c_id] = []
            clusters_b[c_id].append(table)
        
        for table, c_id in partition_r.items():
            if c_id not in clusters_r:
                clusters_r[c_id] = []
            clusters_r[c_id].append(table)
        
        print(f"\nCluster assignments:")
        print(f"{'Table':<30} {'Binary':<20} {'Reinforced':<20}")
        print("-" * 70)
        
        all_tables = set(partition_b.keys()) | set(partition_r.keys())
        for table in sorted(all_tables):
            c_b = partition_b.get(table, "?")
            c_r = partition_r.get(table, "?")
            changed = "✓" if c_b != c_r else ""
            print(f"{table:<30} Cluster {c_b:<19} Cluster {c_r:<19} {changed}")
    
    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph building strategies")
    parser.add_argument(
        "--db-id",
        type=str,
        default="debit_card_specializing",
        help="Database ID to compare",
    )
    args = parser.parse_args()
    
    compare_strategies(args.db_id)


if __name__ == "__main__":
    main()
