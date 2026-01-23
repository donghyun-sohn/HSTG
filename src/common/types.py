#!/usr/bin/env python3
"""
Type definitions for schema and graph structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


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
    tables: Dict[str, TableInfo]  # table_name -> TableInfo
    schema_text: str = ""  # Raw CREATE TABLE statements


@dataclass
class ClusterInfo:
    """Represents a super cluster (super node) in the graph."""
    cluster_id: str
    table_names: List[str]
    indicator_vector: Optional[List[float]] = None
    centroid: Optional[Any] = None
