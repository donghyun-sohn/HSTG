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

    # =========================================================================
    # Step 1: Graph Construction
    # =========================================================================
    # Build hybrid graph for each schema:
    # - Hard Edges: Explicit FK relationships (weight = 1.0 + semantic_sim * 0.5)
    # - Soft Edges: Inferred from column name overlap + semantic similarity
    # - Type Constraint: Only same-type columns can form soft edges
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Graph Construction")
    print("=" * 80)
    print("TODO: Implement graph construction using preprocessed schemas")
    print("  - Use semantic names for embedding generation")
    print("  - Use structural types for type-based edge filtering")
    print("  - Apply Reinforced Hard Edge Strategy")

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
    print("TODO: Implement Louvain clustering on hybrid graph")
    print("  - Detect communities using edge weights")
    print("  - Compute cluster centroid vectors")
    print("  - Save artifacts to data/processed/")

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
