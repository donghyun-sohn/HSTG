#!/usr/bin/env python3
"""
Step 1: Offline Graph Builder

This script runs once during preprocessing.
Reads JSONL and builds graphs per database, saving to files.

Process:
1. Load unique schemas from mini_dev_prompt.jsonl (single pass)
2. For each db_id:
   - Parse DDL into TableNode objects
   - Serialize tables for embedding
   - Generate embeddings (OpenAI API)
   - Build hybrid graph (Hard + Soft edges)
   - Run clustering (Louvain algorithm)
   - Save graph and clusters
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.common.loader import load_unique_schemas
from src.offline.schema_parser import parse_bird_schema
from src.offline.serializer import serialize_schema
from src.offline.embedder import get_embeddings_batch
from src.offline.graph_builder import build_hybrid_graph
from src.offline.clustering import detect_communities

# Get project root (go up from src/offline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"


def main() -> None:
    """Main offline processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build offline graphs and clusters")
    parser.add_argument(
        "--strategy",
        default="reinforced",
        choices=["binary", "reinforced"],
        help="Edge weighting strategy: 'binary' (original) or 'reinforced' (proposed)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 1: Offline Graph Construction")
    print(f"Strategy: {args.strategy}")
    print("=" * 60)
    
    # 1. Load unique schemas
    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        print(f"Please ensure mini_dev-main is in the project root.")
        return
        
    schemas = load_unique_schemas(DATA_FILE)
    
    if not schemas:
        print("No schemas found. Exiting.")
        return
    
    # Create output directories
    (OUTPUT_DIR / "db_graphs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "clusters").mkdir(parents=True, exist_ok=True)

    # 2. Process each database
    for idx, (db_id, schema_data) in enumerate(schemas.items(), 1):
        print(f"\n[{idx}/{len(schemas)}] Processing DB: {db_id}...")
        
        try:
            # Step 1: Parse schema
            schema_info = parse_bird_schema(schema_data["schema_raw"], db_id)
            print(f"  Parsed {len(schema_info.tables)} tables")
            
            # Step 2: Serialize tables for embedding
            serialized = serialize_schema(schema_info)
            print(f"  Serialized {len(serialized)} tables")
            
            # Step 3: Generate embeddings (batch processing)
            texts = list(serialized.values())
            table_names = list(serialized.keys())
            embeddings = get_embeddings_batch(texts)
            vectors = dict(zip(table_names, embeddings))
            print(f"  Generated {len(vectors)} embeddings")
            
            # Step 4: Build hybrid graph with specified strategy
            G = build_hybrid_graph(schema_info.tables, vectors, strategy=args.strategy)
            num_edges = G.number_of_edges()
            num_hard = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "hard")
            num_soft = num_edges - num_hard
            print(f"  Built graph: {G.number_of_nodes()} nodes, {num_edges} edges ({num_hard} hard, {num_soft} soft)")
            
            # Step 5: Clustering
            partition, cluster_vectors = detect_communities(G)
            num_clusters = len(set(partition.values()))
            print(f"  Detected {num_clusters} clusters")
            
            # Step 6: Save artifacts (include strategy in filename for comparison)
            graph_path = OUTPUT_DIR / "db_graphs" / f"{db_id}_{args.strategy}.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
            print(f"  Saved graph to {graph_path}")
            
            # Convert numpy arrays to lists for JSON serialization
            cluster_data = {
                "db_id": db_id,
                "strategy": args.strategy,
                "partition": partition,
                "cluster_vectors": {str(k): v.tolist() for k, v in cluster_vectors.items()},
                "num_clusters": num_clusters,
            }
            
            cluster_path = OUTPUT_DIR / "clusters" / f"{db_id}_{args.strategy}.json"
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
