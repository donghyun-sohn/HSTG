#!/usr/bin/env python3
"""
Test script to verify example data extraction from BIRD DDL.
"""

from __future__ import annotations

from pathlib import Path

from src.common.loader import iter_bird_data
from src.offline.schema_preprocess import parse_bird_schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"


def main() -> None:
    print("=" * 60)
    print("Testing Example Data Extraction from BIRD DDL")
    print("=" * 60)
    print()

    # Get first item
    for item in iter_bird_data(DATA_FILE):
        db_id = item.get("db_id")
        schema_str = item.get("schema", "")

        if schema_str:
            schema_info = parse_bird_schema(schema_str, db_id)

            print(f"Database: {db_id}")
            print("=" * 60)

            for table_name, table_node in schema_info.tables.items():
                print(f"\nTable: {table_name}")
                print(f"Columns: {len(table_node.columns)}")

                for i, (col, comment, examples) in enumerate(
                    zip(table_node.columns, table_node.comments, table_node.column_examples)
                ):
                    print(f"  [{i+1}] {col}")
                    if comment:
                        comment_display = (
                            comment[:60] + "..." if len(comment) > 60 else comment
                        )
                        print(f"      Comment: {comment_display}")
                    if examples:
                        print(f"      Examples: {examples}")

            break

    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    main()
