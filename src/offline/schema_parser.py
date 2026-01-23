#!/usr/bin/env python3
"""
DDL Parser for BIRD schema strings.
Parses CREATE TABLE statements into structured TableNode objects.
"""

from __future__ import annotations

import re
from typing import Dict

from src.common.types import TableNode, SchemaInfo


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
        
        # Parse columns, comments, and constraints
        columns = []
        comments = []
        primary_keys = []
        explicit_fks = []
        
        # Handle multi-line definitions and constraints
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        
        for line in lines:
            line_upper = line.upper()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for PRIMARY KEY
            if "PRIMARY KEY" in line_upper:
                # Extract column names from PRIMARY KEY (col1, col2)
                pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", line_upper)
                if pk_match:
                    pk_cols = [c.strip().replace("`", "") for c in pk_match.group(1).split(",")]
                    primary_keys.extend(pk_cols)
                continue
            
            # Check for FOREIGN KEY constraint
            if "FOREIGN KEY" in line_upper or "CONSTRAINT" in line_upper:
                # Extract referenced table name (case-sensitive from original line)
                fk_match = re.search(
                    r"REFERENCES\s+(?:`)?([a-zA-Z0-9_]+)(?:`)?",
                    line,
                    re.IGNORECASE,
                )
                if fk_match:
                    ref_table = fk_match.group(1)
                    if ref_table not in explicit_fks:
                        explicit_fks.append(ref_table)
                continue
            
            # Parse regular column definition
            # Format: "ColumnName type, -- comment" or "`ColumnName` type, -- comment"
            # Split by '--' to separate definition from comment
            parts = line.split("--", 1)
            col_def = parts[0].strip().rstrip(",")
            
            # Extract column name (first word, remove backticks)
            col_parts = col_def.split()
            if not col_parts:
                continue
            
            col_name = col_parts[0].replace("`", "").strip()
            
            # Skip constraint keywords
            if col_name.upper() in ["PRIMARY", "CONSTRAINT", "FOREIGN", "KEY", "UNIQUE", "CHECK"]:
                continue
            
            columns.append(col_name)
            
            # Extract comment if present
            if len(parts) > 1:
                comment = parts[1].strip()
                comments.append(comment)
            else:
                comments.append("")
        
        tables[table_name] = TableNode(
            name=table_name,
            columns=columns,
            comments=comments,
            raw_ddl=ddl,
            explicit_fks=explicit_fks,
            primary_keys=primary_keys,
        )
    
    return SchemaInfo(
        db_id=db_id,
        tables=tables,
        schema_text=schema_str,
    )
