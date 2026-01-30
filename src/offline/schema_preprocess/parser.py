#!/usr/bin/env python3
"""
DDL Parser for BIRD schema strings.
Parses CREATE TABLE statements into structured TableNode objects.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Any, Tuple

from src.common.types import TableNode, SchemaInfo


def _extract_column_type(col_definition: str) -> str:
    """
    Extract and normalize the data type from a column definition string.
    
    Args:
        col_definition: Column definition like "CustomerID integer" or "`Date` text"
        
    Returns:
        Normalized data type string (lowercase)
    """
    clean_def = col_definition.replace("`", "").strip()
    parts = clean_def.split()
    if len(parts) < 2:
        return "unknown"
    type_str = parts[1].lower()
    type_mapping = {
        "int": "integer", "bigint": "integer", "smallint": "integer", "tinyint": "integer",
        "varchar": "text", "char": "text", "string": "text",
        "float": "real", "double": "real", "decimal": "real", "numeric": "real",
        "boolean": "integer", "bool": "integer",
        "datetime": "text", "timestamp": "text", "date": "text", "time": "text",
    }
    base_type = type_str.split("(")[0]
    return type_mapping.get(base_type, base_type)


def _extract_example_data(comment: str) -> List[Any]:
    """
    Extract example data from BIRD DDL comment.
    
    Handles patterns like:
    - "-- example: [3, 5]"
    - "-- example: ['CZK', 'EUR']"
    - "-- client segment, example: ['SME', 'LAM']"
    
    Args:
        comment: Comment string from DDL
        
    Returns:
        List of example values, or empty list if not found
    """
    if not comment:
        return []
    
    # Look for "example:" followed by a list pattern
    # Pattern: "example:" followed by optional whitespace and then [...]
    example_pattern = r"example:\s*(\[[^\]]*\])"
    match = re.search(example_pattern, comment, re.IGNORECASE)
    
    if not match:
        return []
    
    example_str = match.group(1)
    
    try:
        # Use ast.literal_eval to safely parse the list
        # This handles both [3, 5] and ['CZK', 'EUR'] correctly
        example_list = ast.literal_eval(example_str)
        if isinstance(example_list, list):
            return example_list
        else:
            # If it's a single value, wrap it in a list
            return [example_list]
    except (ValueError, SyntaxError):
        # If parsing fails, try to extract values manually
        # This is a fallback for edge cases
        try:
            # Remove brackets and split by comma
            inner = example_str.strip("[]")
            if not inner:
                return []
            # Try to parse each element
            values = []
            for item in inner.split(","):
                item = item.strip().strip("'\"")
                # Try to convert to number if possible
                try:
                    if "." in item:
                        values.append(float(item))
                    else:
                        values.append(int(item))
                except ValueError:
                    values.append(item)
            return values
        except Exception:
            return []


def parse_bird_schema(schema_str: str, db_id: str) -> SchemaInfo:
    """
    Parse BIRD schema string into structured SchemaInfo.
    
    Args:
        schema_str: Raw CREATE TABLE statements
        db_id: Database identifier
        
    Returns:
        SchemaInfo with parsed tables
    """
    tables: Dict[str, TableNode] = {}
    
    # Split by semicolon to get individual CREATE TABLE statements
    ddl_statements = [s.strip() for s in schema_str.split(";") if s.strip()]
    
    for ddl in ddl_statements:
        if not ddl.upper().startswith("CREATE TABLE"):
            continue
        
        # Extract table name
        table_name_match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`)?([a-zA-Z0-9_]+)(?:`)?",
            ddl,
            re.IGNORECASE,
        )
        if not table_name_match:
            continue
        table_name = table_name_match.group(1)
        
        # Extract body (content between parentheses)
        body_start = ddl.find("(")
        body_end = ddl.rfind(")")
        if body_start == -1 or body_end == -1 or body_start >= body_end:
            continue
        
        body = ddl[body_start + 1 : body_end]
        
        # Parse columns, comments, constraints, column types, and FK details
        columns = []
        comments = []
        column_examples = []
        column_types = []
        primary_keys = []
        explicit_fks = []
        fk_column_refs: Dict[str, Tuple[str, str]] = {}
        
        # Handle multi-line definitions and constraints
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        
        for line in lines:
            line_upper = line.upper()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for PRIMARY KEY
            if "PRIMARY KEY" in line_upper:
                pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", line_upper)
                if pk_match:
                    pk_cols = [c.strip().replace("`", "") for c in pk_match.group(1).split(",")]
                    primary_keys.extend(pk_cols)
                continue
            
            # Check for FOREIGN KEY constraint - extract full (local_col, ref_table, ref_col)
            if "FOREIGN KEY" in line_upper or "CONSTRAINT" in line_upper:
                fk_pattern = r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+(?:`)?(\w+)(?:`)?\s*\(([^)]+)\)"
                fk_match = re.search(fk_pattern, line, re.IGNORECASE)
                if fk_match:
                    local_col = fk_match.group(1).strip().replace("`", "")
                    ref_table = fk_match.group(2).strip().replace("`", "")
                    ref_col = fk_match.group(3).strip().replace("`", "")
                    fk_column_refs[local_col.upper()] = (ref_table, ref_col)
                    if ref_table not in explicit_fks:
                        explicit_fks.append(ref_table)
                continue
            
            # Parse regular column definition
            parts = line.split("--", 1)
            col_def = parts[0].strip().rstrip(",")
            
            col_parts = col_def.split()
            if not col_parts:
                continue
            
            col_name = col_parts[0].replace("`", "").strip()
            
            if col_name.upper() in ["PRIMARY", "CONSTRAINT", "FOREIGN", "KEY", "UNIQUE", "CHECK"]:
                continue
            
            columns.append(col_name)
            column_types.append(_extract_column_type(col_def))
            
            if len(parts) > 1:
                comment = parts[1].strip()
                comments.append(comment)
                column_examples.append(_extract_example_data(comment))
            else:
                comments.append("")
                column_examples.append([])
        
        tables[table_name] = TableNode(
            name=table_name,
            columns=columns,
            comments=comments,
            column_examples=column_examples,
            column_types=column_types,
            raw_ddl=ddl,
            explicit_fks=explicit_fks,
            fk_column_refs=fk_column_refs,
            primary_keys=primary_keys,
        )
    
    return SchemaInfo(
        db_id=db_id,
        tables=tables,
        schema_text=schema_str,
    )
