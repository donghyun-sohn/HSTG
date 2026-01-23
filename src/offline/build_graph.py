#!/usr/bin/env python3
"""
Step 1: Offline Graph Builder

This script runs once during preprocessing.
Reads JSONL and builds graphs per database, saving to files.

Process:
1. Load unique schemas from mini_dev_prompt.jsonl (single pass)
2. For each db_id:
   - Create Vertex with each table names
   - Create Table Indicator Vector
   - Connect edges with weights (PK_FK relationship + semantic relationship)
   - Clustering (semantic similarity, community detection, etc.)
   - Create super cluster's indicator vector
   - Check edges for each super clusters
3. Save graphs and clusters to data/processed/
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.common.loader import load_unique_schemas

# Get project root (go up from src/offline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"


def build_hybrid_graph(schema_data: dict) -> dict:
    """
    [Placeholder] Build Hybrid Graph
    Node: Table, Edge: FK + Semantic
    
    TODO: Implement actual logic
    """
    db_id = schema_data.get("db_id", "unknown")
    print(f"  Building graph for {db_id}...")
    # Placeholder: Actually parse tables, extract FK relationships, embeddings, etc.
    return {"db_id": db_id, "nodes": [], "edges": []}


def run_semantic_clustering(graph: dict) -> dict:
    """
    [Placeholder] Clustering (Create Super Nodes)
    
    TODO: Implement actual logic
    """
    db_id = graph.get("db_id", "unknown")
    print(f"  Clustering for {db_id}...")
    # Placeholder: Actually perform semantic similarity, community detection, etc.
    return {"db_id": db_id, "clusters": []}


def main() -> None:
    """Main offline processing pipeline."""
    print("=" * 60)
    print("Step 1: Offline Graph Construction")
    print("=" * 60)
    
    # 1. Read file and extract unique DB schemas only (ignore questions)
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

    # 2. Build and save graphs for each DB
    for idx, (db_id, schema_data) in enumerate(schemas.items(), 1):
        print(f"\n[{idx}/{len(schemas)}] Processing DB: {db_id}...")
        
        # A. Build Hybrid Graph (Node: Table, Edge: FK + Semantic)
        graph = build_hybrid_graph(schema_data)
        
        # B. Clustering (Create Super Nodes)
        clusters = run_semantic_clustering(graph)
        
        # C. Save (Pickle for graph, JSON for clusters)
        graph_path = OUTPUT_DIR / "db_graphs" / f"{db_id}.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        print(f"  Saved graph to {graph_path}")
            
        cluster_path = OUTPUT_DIR / "clusters" / f"{db_id}.json"
        with open(cluster_path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2, ensure_ascii=False)
        print(f"  Saved clusters to {cluster_path}")
            
    print("\n" + "=" * 60)
    print(f"All offline processing done! Processed {len(schemas)} databases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
