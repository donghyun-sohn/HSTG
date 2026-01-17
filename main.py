#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from input import extract_entities


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    BASE_DIR / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
)


def iter_questions(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="BIRD mini-dev keyword demo")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to mini_dev_prompt.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to show",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "llm", "rule"],
        help="Extraction mode",
    )
    args = parser.parse_args()

    # Step 1: Load schema/SQL/question and extract keywords
    count = 0
    for item in iter_questions(args.data):
        question = item.get("question", "").strip()
        if not question:
            continue
        schema = item.get("schema", "")
        sql = item.get("SQL", "")
        db_id = item.get("db_id", "")

        keywords = extract_entities(question, mode=args.mode)
        print(f"[{item.get('question_id')}] db_id={db_id}")
        print(f"  question: {question}")
        print(f"  schema_loaded: {bool(schema)} (chars={len(schema)})")
        print(f"  sql: {sql}")
        print(f"  keywords: {keywords}")
        print()
        count += 1
        if count >= args.limit:
            break


if __name__ == "__main__":
    main()
