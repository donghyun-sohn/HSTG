#!/usr/bin/env python3
"""
Schema Preprocessing Package

This package handles the transformation of raw DDL schema strings into
structured formats optimized for graph construction and similarity search.

Pipeline:
    JSONL → parser.py → preprocessor.py → serializer.py
              ↓              ↓                ↓
         SchemaInfo    PreprocessedSchema   Embedding-ready text

Main exports:
- parse_bird_schema: Parse DDL into TableNode/SchemaInfo
- preprocess_schema: Transform to semantic names + structural types
- serialize_schema: Legacy serialization for embedding
- serialize_preprocessed_schema: Optimized serialization with FQN/PK/FK/examples
"""

from src.offline.schema_preprocess.parser import (
    parse_bird_schema,
    _extract_example_data,
)

from src.offline.schema_preprocess.preprocessor import (
    preprocess_schema,
    serialize_preprocessed_table,
    serialize_preprocessed_schema,
    get_type_compatible_pairs,
    to_dict,
    PreprocessedColumn,
    PreprocessedTable,
    PreprocessedSchema,
)

from src.offline.schema_preprocess.serializer import (
    serialize_table,
    serialize_schema,
)

__all__ = [
    # Parser
    "parse_bird_schema",
    "_extract_example_data",
    # Preprocessor
    "preprocess_schema",
    "serialize_preprocessed_table",
    "serialize_preprocessed_schema",
    "get_type_compatible_pairs",
    "to_dict",
    "PreprocessedColumn",
    "PreprocessedTable",
    "PreprocessedSchema",
    # Serializer
    "serialize_table",
    "serialize_schema",
]
