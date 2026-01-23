#!/usr/bin/env python3
"""
Type definitions for schema and graph structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TableNode:
    """Represents a database table node with columns, comments, and relationships."""
    name: str
    columns: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)  # Column comments/descriptions
    raw_ddl: str = ""
    explicit_fks: List[str] = field(default_factory=list)  # Referenced table names
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
