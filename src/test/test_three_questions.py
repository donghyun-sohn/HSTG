#!/usr/bin/env python3
"""
Show graph input JSON examples for 3 BIRD questions.
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
    print("Graph Input JSON Examples for 3 BIRD Questions")
    print("=" * 80)
    print()

    # Target question IDs
    target_ids = [1471, 1472, 1473]

    for item in iter_bird_data(DATA_FILE):
        question_id = item.get("question_id")
        if question_id not in target_ids:
            continue

        db_id = item.get("db_id", "")
        question = item.get("question", "")
        evidence = item.get("evidence", "")
        schema_str = item.get("schema", "")

        if not schema_str:
            continue

        print(f"\n{'='*80}")
        print(f"Question ID: {question_id}")
        print(f"Database: {db_id}")
        print(f"Question: {question}")
        if evidence:
            print(f"Evidence: {evidence}")
        print(f"{'='*80}\n")

        # Parse schema
        schema_info = parse_bird_schema(schema_str, db_id)

        # Generate graph input
        try:
            graph_input = generate_graph_input(
                question=question,
                schema_info=schema_info,
                question_id=question_id,
                evidence=evidence,
            )

            # Pretty print the JSON
            print(json.dumps(graph_input, indent=2, ensure_ascii=False))
            print()

            # Print summary
            print("-" * 80)
            print("Summary:")
            print(f"  Graph Anchors ({len(graph_input['graph_anchors'])}):")
            for anchor in graph_input["graph_anchors"]:
                anchor_info = f"    • {anchor['id']} ({anchor['type']})"
                anchor_info += f" - confidence: {anchor['confidence']:.2f}"
                anchor_info += f" - {anchor['hit_type']}"
                if "matched_values" in anchor:
                    anchor_info += f" - matched: {anchor['matched_values']}"
                print(anchor_info)
            print(f"  Intent: {graph_input['traversal_hints']['intent']}")
            print(f"  Operators: {graph_input['traversal_hints']['operators']}")
            print(f"  Semantic Gravity: {graph_input['traversal_hints']['semantic_gravity']}")
            print(f"  Max Hops: {graph_input['traversal_hints']['max_hops']}")
            print("-" * 80)
            print()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print()

        # Stop after processing all target questions
        if len(target_ids) == 0:
            break
        target_ids.remove(question_id)
        if len(target_ids) == 0:
            break

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
