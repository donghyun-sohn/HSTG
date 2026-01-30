#!/usr/bin/env python3
"""
Table serialization for embedding generation.
Converts TableNode objects into text representations for vectorization.

Supports two formats:
1. Legacy: Simple table serialization (Table + Columns + Notes + Examples)
2. Preprocessed: Semantic names with FQN, PK/FK tags, and examples (optimized for similarity)
"""

from __future__ import annotations

from typing import Dict

from src.common.types import TableNode, SchemaInfo, PreprocessedTable


def serialize_table(table: TableNode) -> str:
    """
    Serialize a table into a text representation for embedding.
    
    Format: "Table: [name] | Columns: [col1, col2, ...] | Notes: [comment1; comment2; ...] | Examples: [col1: [ex1, ex2], ...]"
    
    Args:
        table: TableNode to serialize
        
    Returns:
        Text representation of the table
    """
    # Build column list
    columns_str = ", ".join(table.columns)
    
    # Build notes from comments (filter empty comments)
    notes = [c for c in table.comments if c]
    notes_str = "; ".join(notes) if notes else ""
    
    # Build examples string
    examples_parts = []
    for col, examples in zip(table.columns, table.column_examples):
        if examples:
            # Format examples as string representation
            examples_str = str(examples) if len(str(examples)) < 100 else str(examples)[:100] + "..."
            examples_parts.append(f"{col}: {examples_str}")
    
    examples_str = ", ".join(examples_parts) if examples_parts else ""
    
    # Build text representation
    text = f"Table: {table.name} | "
    text += f"Columns: {columns_str} | "
    if notes_str:
        text += f"Notes: {notes_str} | "
    if examples_str:
        text += f"Examples: {examples_str}"
    
    return text


def serialize_schema(schema: SchemaInfo) -> Dict[str, str]:
    """
    Serialize all tables in a schema to text representations.
    
    Args:
        schema: SchemaInfo containing tables
        
    Returns:
        Dict mapping table_name -> serialized text
    """
    return {name: serialize_table(table) for name, table in schema.tables.items()}


# ============================================================================
# Preprocessed Format Serialization (Optimized for Graph Construction)
# ============================================================================

def serialize_preprocessed_table(prep_table: PreprocessedTable) -> str:
    """
    Serialize a preprocessed table for embedding.
    
    This format is optimized for vector similarity search as it includes:
    - Fully Qualified Names (FQN) for each column
    - PK/FK structural annotations
    - Example data from DDL comments
    
    Args:
        prep_table: Preprocessed table with semantic names
        
    Returns:
        String optimized for semantic embedding
    """
    lines = [
        f"Table Context: {prep_table.table_name} ({prep_table.context})",
        "Columns:"
    ]
    
    for sem_name in prep_table.semantic_names:
        lines.append(f"  - {sem_name}")
    
    return "\n".join(lines)


def serialize_preprocessed_schema(tables: Dict[str, PreprocessedTable]) -> Dict[str, str]:
    """
    Serialize all preprocessed tables.
    
    Args:
        tables: Dict of table_name -> PreprocessedTable
        
    Returns:
        Dict mapping table_name -> serialized text for embedding
    """
    return {
        table_name: serialize_preprocessed_table(prep_table)
        for table_name, prep_table in tables.items()
    }
