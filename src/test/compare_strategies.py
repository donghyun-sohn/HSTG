#!/usr/bin/env python3
"""
Inspect graph and clusters (legacy: previously compared binary vs reinforced).
Now uses unified graph from PreprocessedSchema.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx

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


def inspect_graph(db_id: str) -> None:
    """Show graph structure and clusters."""
    print("=" * 70)
    print(f"Graph & Clusters: {db_id}")
    print("=" * 70)

    G = load_graph(db_id)
    clusters = load_clusters(db_id)

    if not G:
        print("Error: Graph not found. Run 'python -m src.test.test_bird_mini' first.")
        return

    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    hard = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "hard")
    soft = G.number_of_edges() - hard
    print(f"  Hard: {hard}, Soft: {soft}")

    print("\nEdges (with join_keys):")
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0)
        t = d.get("type", "?")
        keys = d.get("join_keys", [])
        key_str = ", ".join(f"{k['source_col']}->{k['target_col']}" for k in keys[:2])
        if len(keys) > 2:
            key_str += "..."
        print(f"  {u} <-> {v} [{t}, w={w:.3f}] {key_str}")

    if clusters:
        print(f"\nClusters: {clusters.get('num_clusters', 0)}")
        for table, cid in sorted(clusters.get("partition", {}).items()):
            print(f"  {table}: cluster {cid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect graph and clusters")
    parser.add_argument("--db", "--db-id", dest="db_id", default="debit_card_specializing")
    args = parser.parse_args()
    inspect_graph(args.db_id)


if __name__ == "__main__":
    main()
