#!/usr/bin/env python3
"""
Step 2: Online Inference (Query Processing)

When a question arrives, load pre-built graphs and:
Extract keywords → Match Super Clusters → Filter tables → Generate SQL
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from src.common.loader import iter_bird_data
from src.online.entity_extractor import extract_entities

# Get project root (go up from src/online/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"


def load_graph_for_db(db_id: str) -> dict | None:
    """
    Load graph for specific DB when needed (can be cached).
    
    Args:
        db_id: Database identifier
        
    Returns:
        Graph dict or None if not found
    """
    graph_path = PROCESSED_DIR / "db_graphs" / f"{db_id}.pkl"
    if not graph_path.exists():
        return None
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def load_clusters_for_db(db_id: str) -> dict | None:
    """
    Load cluster information for specific DB.
    
    Args:
        db_id: Database identifier
        
    Returns:
        Cluster dict or None if not found
    """
    cluster_path = PROCESSED_DIR / "clusters" / f"{db_id}.json"
    if not cluster_path.exists():
        return None
    with open(cluster_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online query processing with hierarchical schema linking"
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to mini_dev_prompt.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to process",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "llm", "rule"],
        help="Entity extraction mode",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please ensure mini_dev-main is in the project root.")
        return

    print("=" * 60)
    print("Step 2: Online Query Processing")
    print("=" * 60)
    print()

    count = 0
    
    # [Core] Process JSONL line by line (Online Streaming)
    for item in iter_bird_data(data_path):
        if count >= args.limit:
            break
            
        question = item.get("question", "").strip()
        db_id = item.get("db_id", "")
        question_id = item.get("question_id", "?")
        
        if not question or not db_id:
            continue

        # 1. Load pre-built graph for the DB matching this question
        # (Can be optimized by caching in memory to avoid repeated loads)
        graph = load_graph_for_db(db_id)
        clusters = load_clusters_for_db(db_id)
        
        if not graph:
            print(f"[{question_id}] Skipping {db_id}: Graph not found")
            print(f"  (Run 'python -m src.offline.build_graph' first)")
            print()
            continue

        # 2. Execute Online Pipeline
        print(f"[{question_id}] Question: {question}")
        print(f"  Target DB: {db_id}")
        
        # A. Keyword Extraction
        keywords = extract_entities(question, mode=args.mode)
        print(f"  Keywords: {keywords}")
        
        # B. Retrieval & Routing (Implement novelty here)
        # TODO: find_path_on_graph(graph, keywords)
        # TODO: match_super_clusters(clusters, keywords)
        # TODO: filter_tables_within_clusters(...)
        
        if clusters:
            print(f"  Clusters loaded: {len(clusters.get('clusters', []))} super clusters")
        else:
            print(f"  Clusters: Not found (run test_bird_mini first)")
        
        print("-" * 60)
        print()
        count += 1

    print(f"Processed {count} questions.")


if __name__ == "__main__":
    main()
