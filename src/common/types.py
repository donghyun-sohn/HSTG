#!/usr/bin/env python3
"""
Type definitions for schema and graph structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class TableNode:
    """Represents a database table node with columns, comments, and relationships."""
    name: str
    columns: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)  # Column comments/descriptions
    column_examples: List[List[Any]] = field(default_factory=list)  # Example data for each column
    column_types: List[str] = field(default_factory=list)  # Data type per column (same index as columns)
    raw_ddl: str = ""
    explicit_fks: List[str] = field(default_factory=list)  # Referenced table names
    fk_column_refs: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # col_upper -> (ref_table, ref_col)
    primary_keys: List[str] = field(default_factory=list)


@dataclass
class TableInfo:
    """Represents a database table with its columns and metadata."""
    name: str
    columns: List[str]
    primary_keys: List[str] = None
    foreign_keys: Dict[str, str] = None  # column -> referenced_table.column


@dataclass
class SchemaInfo:
    """Represents a complete database schema."""
    db_id: str
    tables: Dict[str, TableNode]  # table_name -> TableNode
    schema_text: str = ""  # Raw CREATE TABLE statements


@dataclass
class ClusterInfo:
    """Represents a super cluster (super node) in the graph."""
    cluster_id: str
    table_names: List[str]
    indicator_vector: Optional[List[float]] = None
    centroid: Optional[Any] = None


@dataclass
class PreprocessedColumn:
    """
    Represents a preprocessed column with metadata for graph construction.
    
    Used for:
    - Generating semantic names (embedding target)
    - Storing structural type (hard constraint filter)
    - fk_points_to_pk: True if FK references a PK (higher graph weight in future)
    """
    table_name: str
    column_name: str
    data_type: str
    is_primary_key: bool = False
    foreign_key_ref: Optional[str] = None  # Format: "target_table.target_column"
    fk_points_to_pk: bool = True  # True when FK targets a PK column
    examples: List[Any] = field(default_factory=list)


@dataclass
class PreprocessedTable:
    """
    Represents a preprocessed table optimized for graph construction.
    
    Contains:
    - semantic_names: FQN with PK/FK annotations and examples (for embedding)
    - structural_types: Column data types (for hard constraints)
    """
    table_name: str
    context: str  # Brief description of table's domain/purpose
    semantic_names: List[str] = field(default_factory=list)
    structural_types: List[str] = field(default_factory=list)
    columns: List['PreprocessedColumn'] = field(default_factory=list)
