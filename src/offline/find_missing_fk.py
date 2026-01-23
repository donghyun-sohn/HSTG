#!/usr/bin/env python3
"""
Find missing PK-FK relationships by comparing SQL queries with schema definitions.

For each question:
1. Extract JOIN relationships from SQL
2. Extract explicit FK relationships from schema
3. Identify joins that are not defined in schema
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.common.loader import iter_bird_data
from src.offline.schema_parser import parse_bird_schema

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/missing_fk_relationships.json"


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from SQL query."""
    # Normalize SQL (remove comments, extra whitespace)
    sql_clean = re.sub(r"--.*", "", sql)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)
    sql_clean = " ".join(sql_clean.split())
    
    # Find all table references
    tables = set()
    
    # FROM clause
    from_match = re.search(r"\bFROM\s+([^\s(,]+)", sql_clean, re.IGNORECASE)
    if from_match:
        tables.add(from_match.group(1).strip("`"))
    
    # JOIN clauses
    join_pattern = r"\b(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+([^\s(,]+)"
    for match in re.finditer(join_pattern, sql_clean, re.IGNORECASE):
        tables.add(match.group(1).strip("`"))
    
    # UPDATE clause
    update_match = re.search(r"\bUPDATE\s+([^\s(,]+)", sql_clean, re.IGNORECASE)
    if update_match:
        tables.add(update_match.group(1).strip("`"))
    
    return tables


def extract_join_relationships(sql: str) -> List[Tuple[str, str, str]]:
    """
    Extract join relationships from SQL.
    Returns list of (table1, table2, join_column) tuples.
    """
    relationships = []
    
    # Normalize SQL
    sql_clean = re.sub(r"--.*", "", sql)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)
    sql_clean = " ".join(sql_clean.split())
    
    # Build alias map: T1 -> customers, T2 -> yearmonth, etc.
    alias_map = {}
    
    # Extract FROM clause with aliases: FROM table AS alias or FROM table alias
    from_match = re.search(r"\bFROM\s+([^\s(,]+)(?:\s+AS\s+)?([a-zA-Z0-9_]+)?", sql_clean, re.IGNORECASE)
    if from_match:
        table = from_match.group(1).strip("`")
        alias = from_match.group(2).strip("`") if from_match.group(2) else None
        if alias:
            alias_map[alias.upper()] = table
        alias_map[table.upper()] = table
    
    # Extract JOIN clauses with aliases
    join_pattern = r"(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+([^\s(,]+)(?:\s+AS\s+)?([a-zA-Z0-9_]+)?"
    for match in re.finditer(join_pattern, sql_clean, re.IGNORECASE):
        table = match.group(1).strip("`")
        alias = match.group(2).strip("`") if match.group(2) else None
        if alias:
            alias_map[alias.upper()] = table
        alias_map[table.upper()] = table
    
    # Pattern 1: JOIN ... ON alias1.col = alias2.col
    join_on_pattern = r"(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+([^\s(,]+)(?:\s+AS\s+)?([a-zA-Z0-9_]+)?\s+ON\s+([^\s=]+)\s*=\s*([^\s,)]+)"
    for match in re.finditer(join_on_pattern, sql_clean, re.IGNORECASE):
        table2_raw = match.group(1).strip("`")
        alias2 = match.group(2).strip("`") if match.group(2) else None
        col1_expr = match.group(3).strip()
        col2_expr = match.group(4).strip()
        
        # Resolve table2
        table2 = alias_map.get((alias2 or table2_raw).upper(), table2_raw)
        
        # Extract alias/table and column from expressions
        t1_match = re.search(r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", col1_expr, re.IGNORECASE)
        t2_match = re.search(r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", col2_expr, re.IGNORECASE)
        
        if t1_match and t2_match:
            alias1 = t1_match.group(1).strip("`").upper()
            col1 = t1_match.group(2).strip("`")
            alias2_from_expr = t2_match.group(1).strip("`").upper()
            col2 = t2_match.group(2).strip("`")
            
            # Resolve table1
            table1 = alias_map.get(alias1, alias1)
            
            # Verify columns match and get column name
            if col1.lower() == col2.lower():
                relationships.append((table1, table2, col1))
    
    # Pattern 2: WHERE alias1.col = alias2.col (implicit join)
    where_pattern = r"WHERE\s+.*?([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)"
    for match in re.finditer(where_pattern, sql_clean, re.IGNORECASE):
        alias1 = match.group(1).strip("`").upper()
        col1 = match.group(2).strip("`")
        alias2 = match.group(3).strip("`").upper()
        col2 = match.group(4).strip("`")
        
        table1 = alias_map.get(alias1, alias1)
        table2 = alias_map.get(alias2, alias2)
        
        if col1.lower() == col2.lower() and table1.lower() != table2.lower():
            relationships.append((table1, table2, col1))
    
    # Deduplicate and normalize
    normalized = []
    seen = set()
    for t1, t2, col in relationships:
        # Normalize: always put smaller table name first
        pair = tuple(sorted([t1.lower(), t2.lower()])) + (col.lower(),)
        if pair not in seen:
            seen.add(pair)
            normalized.append((t1, t2, col))
    
    return normalized


def find_missing_fk_relationships() -> Dict:
    """Main function to find missing FK relationships."""
    results = defaultdict(lambda: {
        "explicit_fk": [],
        "missing_fk": [],
        "questions": [],
    })
    
    print("Analyzing SQL queries and schemas...")
    
    for item in iter_bird_data(DATA_FILE):
        db_id = item.get("db_id")
        question_id = item.get("question_id")
        sql = item.get("SQL", "")
        schema_str = item.get("schema", "")
        
        if not db_id or not sql or not schema_str:
            continue
        
        # Parse schema to get explicit FK relationships
        schema_info = parse_bird_schema(schema_str, db_id)
        explicit_fks = set()
        
        for table_name, table_node in schema_info.tables.items():
            for ref_table in table_node.explicit_fks:
                # Normalize: always put smaller table name first
                pair = tuple(sorted([table_name.lower(), ref_table.lower()]))
                explicit_fks.add(pair)
        
        # Extract join relationships from SQL
        sql_joins = extract_join_relationships(sql)
        sql_join_pairs = set()
        
        for t1, t2, col in sql_joins:
            pair = tuple(sorted([t1.lower(), t2.lower()]))
            sql_join_pairs.add((pair, col.lower()))
        
        # Find missing relationships
        missing = []
        for (t1, t2), col in sql_join_pairs:
            # Skip self-joins (same table)
            if t1 == t2:
                continue
                
            pair = (t1, t2)
            if pair not in explicit_fks:
                # Find original table names (case-sensitive)
                original_t1 = None
                original_t2 = None
                for table_name in schema_info.tables.keys():
                    if table_name.lower() == t1:
                        original_t1 = table_name
                    if table_name.lower() == t2:
                        original_t2 = table_name
                
                if original_t1 and original_t2:
                    missing.append({
                        "table1": original_t1,
                        "table2": original_t2,
                        "column": col,
                        "question_id": question_id,
                    })
        
        # Store results
        if explicit_fks or missing:
            # Convert explicit_fks to readable format
            explicit_list = []
            for t1, t2 in sorted(explicit_fks):
                # Find original table names
                orig_t1 = None
                orig_t2 = None
                for table_name in schema_info.tables.keys():
                    if table_name.lower() == t1:
                        orig_t1 = table_name
                    if table_name.lower() == t2:
                        orig_t2 = table_name
                if orig_t1 and orig_t2:
                    explicit_list.append({"table1": orig_t1, "table2": orig_t2})
            
            results[db_id]["explicit_fk"] = explicit_list
            if missing:
                results[db_id]["missing_fk"].extend(missing)
                results[db_id]["questions"].append(question_id)
    
    # Convert to regular dict and format
    output = {}
    for db_id, data in results.items():
        # Count unique missing relationships
        from collections import Counter
        unique_missing = Counter()
        for rel in data["missing_fk"]:
            key = (rel["table1"], rel["table2"], rel["column"])
            unique_missing[key] += 1
        
        # Format unique missing relationships
        missing_unique = []
        for (t1, t2, col), count in unique_missing.items():
            missing_unique.append({
                "table1": t1,
                "table2": t2,
                "column": col,
                "usage_count": count,
            })
        
        output[db_id] = {
            "explicit_fk": data["explicit_fk"],
            "missing_fk_unique": sorted(missing_unique, key=lambda x: x["usage_count"], reverse=True),
            "missing_fk_all": data["missing_fk"],  # Keep all for reference
            "num_questions_with_missing": len(set(data["questions"])),
            "num_unique_missing": len(unique_missing),
        }
    
    return output


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Finding Missing PK-FK Relationships")
    print("=" * 60)
    print()
    
    results = find_missing_fk_relationships()
    
    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nSummary:")
    print("-" * 60)
    
    total_missing = 0
    for db_id, data in results.items():
        num_missing = data.get("num_unique_missing", 0)
        num_explicit = len(data["explicit_fk"])
        if num_missing > 0:
            print(f"{db_id}:")
            print(f"  Explicit FK: {num_explicit}")
            print(f"  Missing FK: {num_missing} unique relationships (from {data['num_questions_with_missing']} questions)")
            total_missing += num_missing
    
    print(f"\nTotal missing FK relationships found: {total_missing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
