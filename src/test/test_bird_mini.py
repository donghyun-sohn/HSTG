#!/usr/bin/env python3
"""
Main test script for BIRD mini-dev benchmark.

This script demonstrates the complete HSTG pipeline:
- Step 0: Schema preprocessing (Semantic Names + Structural Types)
- Step 1: Graph construction (Hybrid edges: Hard FK + Soft semantic)
- Step 2: Clustering (Louvain community detection → Super Nodes)
- Step 3: Online query processing (Entity extraction → Schema linking → Graph traversal)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Any

from src.common.loader import load_unique_schemas
from src.offline.schema_preprocess import (
    parse_bird_schema,
    preprocess_schema,
    serialize_preprocessed_schema,
    to_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"


def main() -> None:
    print("=" * 80)
    print("HSTG Pipeline Test - BIRD mini-dev")
    print("=" * 80)

    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found")
        print("Please ensure mini_dev-main is in the project root")
        return

    # =========================================================================
    # Step 0: Schema Preprocessing
    # =========================================================================
    # Load JSONL and preprocess all unique schemas into:
    # - Semantic Names: FQN with PK/FK annotations and example data (for embedding)
    # - Structural Types: Column data types (for hard type constraints)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 0: Schema Preprocessing")
    print("=" * 80)

    schemas = load_unique_schemas(str(DATA_FILE))
    print(f"Loaded {len(schemas)} unique databases\n")

    preprocessed_schemas: Dict[str, Any] = {}

    for db_id, schema_data in schemas.items():
        # Parse DDL into SchemaInfo
        parsed_schema = parse_bird_schema(schema_data["schema_raw"], db_id)
        
        # Preprocess into semantic names + structural types
        preprocessed = preprocess_schema(parsed_schema)
        preprocessed_schemas[db_id] = preprocessed

        # Summary
        num_tables = len(preprocessed.tables)
        num_fks = len(preprocessed.fk_mapping)
        print(f"  [{db_id}] {num_tables} tables, {num_fks} FK relationships")

    print(f"\nStep 0 complete: Preprocessed {len(preprocessed_schemas)} schemas")

    graphs: Dict[str, Any] = {}

    # =========================================================================
    # Step 1: Graph Construction
    # =========================================================================
    # Build hybrid graph for each schema:
    # - Node: Table with column-level vectors (semantic_names embedded)
    # - Hard Edges: fk_mapping, W_hard = 1.0 + Sim_vec * 0.5
    # - Soft Edges: Type-compatible columns, W_soft = Sim_name*0.6 + Sim_vec*0.4
    # - Edge metadata: source_col, target_col for JOIN key mapping
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Graph Construction")
    print("=" * 80)

    try:
        from src.offline.graph_builder import build_hybrid_graph

        for db_id, preprocessed in preprocessed_schemas.items():
            G = build_hybrid_graph(preprocessed)
            graphs[db_id] = G
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            hard_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "hard")
            soft_edges = num_edges - hard_edges
            print(f"  [{db_id}] {num_nodes} nodes, {num_edges} edges ({hard_edges} hard, {soft_edges} soft)")
        print(f"\nStep 1 complete: Built graphs for {len(graphs)} databases")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Step 2: Clustering (Super Node Creation)
    # =========================================================================
    # Run Louvain community detection on hybrid graph:
    # - Group related tables into Super Nodes
    # - Compute centroid vector for each cluster
    # - Save graph and cluster artifacts
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Clustering (Super Node Creation)")
    print("=" * 80)

    if not graphs:
        print("  Skipped: No graphs from Step 1")
    else:
        try:
            from src.offline.clustering import detect_communities

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "db_graphs").mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "clusters").mkdir(parents=True, exist_ok=True)

            for db_id, G in graphs.items():
                partition, cluster_vectors = detect_communities(G)
                num_clusters = len(set(partition.values()))

                graph_path = OUTPUT_DIR / "db_graphs" / f"{db_id}.pkl"
                with open(graph_path, "wb") as f:
                    pickle.dump(G, f)

                cluster_data = {
                    "db_id": db_id,
                    "partition": partition,
                    "cluster_vectors": {str(k): v.tolist() for k, v in cluster_vectors.items()},
                    "num_clusters": num_clusters,
                }
                cluster_path = OUTPUT_DIR / "clusters" / f"{db_id}.json"
                with open(cluster_path, "w", encoding="utf-8") as f:
                    json.dump(cluster_data, f, indent=2, ensure_ascii=False)

                print(f"  [{db_id}] {num_clusters} clusters, saved to {graph_path.name}, {cluster_path.name}")
            print(f"\nStep 2 complete: Saved graphs and clusters for {len(graphs)} databases")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Step 3: Online Query Processing
    # =========================================================================
    # For each question:
    # 1. Extract entities (concepts, values, operations) using LLM
    # 2. Schema linking: Match entities to tables/columns
    # 3. Cluster routing: Find relevant Super Nodes via semantic similarity
    # 4. Graph traversal: Navigate from anchors to related tables
    # 5. Output: List of relevant tables for SQL generation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Online Query Processing")
    print("=" * 80)
    print("TODO: Implement online query processing")
    print("  - Entity extraction with table name context")
    print("  - Schema linking using preprocessed semantic names")
    print("  - Cluster routing via semantic gravity")
    print("  - Graph traversal from anchors")

    # =========================================================================
    # Step 4: Evaluation
    # =========================================================================
    # Compare predicted tables with ground truth SQL:
    # - Precision: How many predicted tables are correct?
    # - Recall: How many required tables were found?
    # - F1 Score
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Evaluation")
    print("=" * 80)
    print("TODO: Implement evaluation metrics")
    print("  - Extract tables from ground truth SQL")
    print("  - Compare with predicted tables from graph traversal")
    print("  - Calculate precision, recall, F1")

    print("\n" + "=" * 80)
    print("Pipeline test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
