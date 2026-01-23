#!/usr/bin/env python3
"""
Table serialization for embedding generation.
Converts TableNode objects into text representations for vectorization.
"""

from __future__ import annotations

from src.common.types import TableNode, SchemaInfo


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
