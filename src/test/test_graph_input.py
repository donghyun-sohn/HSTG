#!/usr/bin/env python3
"""
Test script for graph input generation pipeline.
Tests the complete flow: entity extraction -> schema linking -> graph input JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.loader import iter_bird_data
from src.offline.schema_preprocess import parse_bird_schema
from src.online.graph_input_generator import generate_graph_input

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"


def main() -> None:
    print("=" * 80)
    print("Testing Graph Input Generation Pipeline")
    print("=" * 80)
    print()

    # Find question ID 1471 (the example from the requirements)
    target_id = 1471

    for item in iter_bird_data(DATA_FILE):
        question_id = item.get("question_id")
        if question_id != target_id:
            continue

        db_id = item.get("db_id", "")
        question = item.get("question", "")
        evidence = item.get("evidence", "")
        schema_str = item.get("schema", "")

        if not schema_str:
            print("No schema found!")
            return

        print(f"Question ID: {question_id}")
        print(f"Database: {db_id}")
        print(f"Question: {question}")
        print(f"Evidence: {evidence}")
        print()

        # Parse schema
        schema_info = parse_bird_schema(schema_str, db_id)
        print(f"Parsed schema: {len(schema_info.tables)} tables")
        print()

        # Generate graph input
        print("Generating graph input JSON...")
        print("-" * 80)
        try:
            graph_input = generate_graph_input(
                question=question,
                schema_info=schema_info,
                question_id=question_id,
                evidence=evidence,
            )

            # Pretty print the result
            print(json.dumps(graph_input, indent=2, ensure_ascii=False))
            print()

            # Print summary
            print("=" * 80)
            print("Summary:")
            print(f"  - Graph Anchors: {len(graph_input['graph_anchors'])}")
            for anchor in graph_input["graph_anchors"]:
                print(f"    * {anchor['id']} ({anchor['type']}) - confidence: {anchor['confidence']:.2f} - {anchor['hit_type']}")
            print(f"  - Intent: {graph_input['traversal_hints']['intent']}")
            print(f"  - Operators: {graph_input['traversal_hints']['operators']}")
            print(f"  - Semantic Gravity: {graph_input['traversal_hints']['semantic_gravity']}")
            print("=" * 80)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        break

    print("\nTest completed!")


if __name__ == "__main__":
    main()
