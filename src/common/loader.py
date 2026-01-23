#!/usr/bin/env python3
"""
Efficient data loader for BIRD mini-dev dataset.
Single-pass reading with generator pattern for memory efficiency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Dict, Any


def iter_bird_data(file_path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Read JSONL file line by line and yield dicts (Generator pattern).
    Memory-efficient for large files.
    
    Args:
        file_path: Path to mini_dev_prompt.jsonl or similar JSONL file
        
    Yields:
        Dict containing question, schema, SQL, db_id, etc.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue


def load_unique_schemas(file_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    [For Offline stage]
    Scan file once to remove duplicate db_ids and return
    a dictionary mapping db_id -> schema info.
    
    Discards questions and only keeps schema structure.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Dict mapping db_id -> schema info dict
    """
    unique_schemas = {}
    path = Path(file_path)
    print(f"Loading schemas from {path}...")
    
    for item in iter_bird_data(path):
        db_id = item.get("db_id")
        if not db_id:
            continue
            
        # Skip if already processed (deduplication)
        if db_id not in unique_schemas:
            # Assume schema info is stored as text in BIRD JSONL
            schema_info = item.get("schema") or item.get("schema_sequence", "")
            
            unique_schemas[db_id] = {
                "db_id": db_id,
                "schema_raw": schema_info,
                # Can add sqlite path info here if needed
            }
    
    print(f"Found {len(unique_schemas)} unique databases.")
    return unique_schemas
