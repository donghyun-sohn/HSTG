#!/usr/bin/env python3
"""
Step 1: Offline Graph Builder

Reads JSONL and builds graphs per database, saving to files.

Process:
1. Load unique schemas from mini_dev_prompt.jsonl
2. For each db_id:
   - Parse DDL and preprocess (semantic names + structural types)
   - Embed column-level semantic names (Ollama)
   - Build hybrid graph (Hard FK + Soft semantic edges)
   - Run clustering (Louvain algorithm)
   - Save graph and clusters
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.common.loader import load_unique_schemas
from src.offline.schema_preprocess import parse_bird_schema, preprocess_schema
from src.offline.graph_builder import build_hybrid_graph
from src.offline.clustering import detect_communities

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"


def main() -> None:
    """Main offline processing pipeline."""
    print("=" * 60)
    print("Step 1: Offline Graph Construction")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        print("Please ensure mini_dev-main is in the project root.")
        return

    schemas = load_unique_schemas(DATA_FILE)
    if not schemas:
        print("No schemas found. Exiting.")
        return

    (OUTPUT_DIR / "db_graphs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "clusters").mkdir(parents=True, exist_ok=True)

    for idx, (db_id, schema_data) in enumerate(schemas.items(), 1):
        print(f"\n[{idx}/{len(schemas)}] Processing DB: {db_id}...")

        try:
            parsed = parse_bird_schema(schema_data["schema_raw"], db_id)
            prep_schema = preprocess_schema(parsed)
            print(f"  Preprocessed {len(prep_schema.tables)} tables")

            G = build_hybrid_graph(prep_schema)
            num_edges = G.number_of_edges()
            num_hard = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "hard")
            num_soft = num_edges - num_hard
            print(f"  Built graph: {G.number_of_nodes()} nodes, {num_edges} edges ({num_hard} hard, {num_soft} soft)")

            partition, cluster_vectors = detect_communities(G)
            num_clusters = len(set(partition.values()))
            print(f"  Detected {num_clusters} clusters")

            graph_path = OUTPUT_DIR / "db_graphs" / f"{db_id}.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
            print(f"  Saved graph to {graph_path}")

            cluster_data = {
                "db_id": db_id,
                "partition": partition,
                "cluster_vectors": {str(k): v.tolist() for k, v in cluster_vectors.items()},
                "num_clusters": num_clusters,
            }
            cluster_path = OUTPUT_DIR / "clusters" / f"{db_id}.json"
            with open(cluster_path, "w", encoding="utf-8") as f:
                json.dump(cluster_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved clusters to {cluster_path}")

        except Exception as e:
            print(f"  Error processing {db_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print(f"All offline processing done! Processed {len(schemas)} databases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
