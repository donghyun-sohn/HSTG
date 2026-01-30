#!/usr/bin/env python3
"""
Test script for schema preprocessing.

Demonstrates the transformation of BIRD schema into:
1. Semantic Names (for embedding)
2. Structural Types (for hard constraints)
"""

import json
from pathlib import Path

from src.common.loader import load_unique_schemas
from src.offline.schema_preprocess import (
    parse_bird_schema,
    preprocess_schema,
    serialize_preprocessed_table,
    to_dict,
    get_type_compatible_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"


def main():
    # Load schemas from BIRD mini-dev
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found")
        print("Please ensure mini_dev-main is in the project root")
        return

    print("=" * 80)
    print("SCHEMA PREPROCESSING DEMO")
    print("=" * 80)

    schemas = load_unique_schemas(str(DATA_FILE))
    num_schemas = len(schemas)

    for i, (db_id, schema_info) in enumerate(schemas.items(), 1):
        print(f"\n{'='*80}")
        print(f"DATABASE: {db_id} [{i}/{num_schemas}]")
        print("=" * 80)
        
        # Parse schema
        parsed_schema = parse_bird_schema(schema_info["schema_raw"], db_id)
        
        # Preprocess schema
        preprocessed = preprocess_schema(parsed_schema)
        
        # Display results for each table
        for table_name, prep_table in preprocessed.tables.items():
            print(f"\n{'─'*60}")
            print(f"Table: {table_name}")
            print(f"Context: {prep_table.context}")
            print("─" * 60)
            
            print("\n① Semantic Names (Embedding Target):")
            print(json.dumps(prep_table.semantic_names, indent=2, ensure_ascii=False))
            
            print("\n② Structural Types (Constraint Vector):")
            print(json.dumps(prep_table.structural_types, indent=2))
            
            print("\n③ Serialized for Embedding:")
            print("-" * 40)
            print(serialize_preprocessed_table(prep_table))
        
        # Show FK mapping
        if preprocessed.fk_mapping:
            print(f"\n{'─'*60}")
            print("FK Mappings (for Hard Edge Creation):")
            print("─" * 60)
            for (src_table, src_col), (dst_table, dst_col) in preprocessed.fk_mapping.items():
                print(f"  {src_table}.{src_col} -> {dst_table}.{dst_col}")
        
        # Show type-compatible pairs (potential soft edges)
        compatible = get_type_compatible_pairs(preprocessed)
        if compatible:
            print(f"\n{'─'*60}")
            print(f"Type-Compatible Pairs (Soft Edge Candidates): {len(compatible)} pairs")
            print("─" * 60)
            # Show first 10 pairs
            for col1, type1, col2, type2 in compatible[:10]:
                print(f"  [{type1}] {col1} <-> {col2}")
            if len(compatible) > 10:
                print(f"  ... and {len(compatible) - 10} more pairs")
        
        # Show full dict format
        print(f"\n{'─'*60}")
        print("Full Preprocessed Schema (Dict Format):")
        print("─" * 60)
        schema_dict = to_dict(preprocessed)
        print(json.dumps(schema_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
