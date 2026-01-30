#!/usr/bin/env python3
"""
Simple test script to extract keywords from first 3 questions using Ollama.
No graph required - just keyword extraction.
"""

from __future__ import annotations

from pathlib import Path

from src.common.loader import iter_bird_data, extract_table_names
from src.online.entity_extractor import extract_entities

# Get project root (go up from src/test/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"


def main() -> None:
    print("=" * 60)
    print("Keyword Extraction Test (First 3 questions)")
    print("=" * 60)
    print()

    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        print("Please ensure mini_dev-main is in the project root.")
        return

    count = 0
    limit = 3

    for item in iter_bird_data(DATA_FILE):
        if count >= limit:
            break

        question = item.get("question", "").strip()
        db_id = item.get("db_id", "")
        question_id = item.get("question_id", "?")
        schema_str = item.get("schema", "")

        if not question or not db_id:
            continue

        # Extract table names from schema
        table_names = []
        if schema_str:
            table_names = extract_table_names(schema_str)

        print(f"[{question_id}] DB: {db_id}")
        print(f"Question: {question}")
        if table_names:
            print(f"Available tables: {', '.join(table_names)}")
        print(f"Extracting keywords with Ollama...")

        try:
            keywords = extract_entities(question, mode="llm", table_names=table_names)
            print(f"Keywords: {keywords}")
        except Exception as e:
            print(f"Error: {e}")
            print("Falling back to rule-based extraction...")
            keywords = extract_entities(question, mode="rule")
            print(f"Keywords (rule-based): {keywords}")

        print("-" * 60)
        print()
        count += 1

    print(f"Processed {count} questions.")


if __name__ == "__main__":
    main()
