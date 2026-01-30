#!/usr/bin/env python3
"""
Schema Linking & Anchor Scoring for hierarchical schema linking.
Matches extracted entities with schema metadata to identify anchors.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
from src.common.types import SchemaInfo, TableNode


def match_values_against_examples(
    values: List[str], schema_info: SchemaInfo
) -> List[Dict[str, Any]]:
    """
    Match extracted values against column example data.
    
    Args:
        values: List of extracted values (e.g., ["EUR", "CZK"])
        schema_info: Parsed schema information
        
    Returns:
        List of matched anchors with confidence scores
    """
    anchors = []
    
    for table_name, table_node in schema_info.tables.items():
        for col_idx, (col_name, examples) in enumerate(
            zip(table_node.columns, table_node.column_examples)
        ):
            if not examples:
                continue
            
            # Check if any extracted value matches the examples
            matched_values = []
            for value in values:
                value_matched = False
                
                # Try numeric matching first (for years, IDs, amounts, etc.)
                try:
                    # Attempt to convert value to number
                    value_num = float(value) if "." in str(value) else int(value)
                    
                    for example in examples:
                        try:
                            # Try to convert example to number
                            example_num = float(example) if "." in str(example) else int(example)
                            if abs(value_num - example_num) < 1e-6:  # Handle floating point precision
                                matched_values.append(value)
                                value_matched = True
                                break
                        except (ValueError, TypeError):
                            continue
                except (ValueError, TypeError):
                    pass
                
                # If numeric matching failed, try string matching
                if not value_matched:
                    value_normalized = str(value).strip().upper()
                    for example in examples:
                        example_normalized = str(example).strip().upper()
                        if value_normalized == example_normalized:
                            matched_values.append(value)
                            break
            
            if matched_values:
                # Calculate confidence based on match ratio
                confidence = len(matched_values) / len(values) if values else 0.0
                anchors.append(
                    {
                        "id": f"{table_name}.{col_name}",
                        "type": "column",
                        "confidence": min(confidence, 1.0),
                        "hit_type": "value_match",
                        "matched_values": matched_values,
                        "table": table_name,
                        "column": col_name,
                    }
                )
    
    return anchors


def match_concepts_against_tables(
    concepts: List[str], schema_info: SchemaInfo
) -> List[Dict[str, Any]]:
    """
    Match extracted concepts against table names.
    
    Args:
        concepts: List of extracted concepts (e.g., ["customers", "transaction"])
        schema_info: Parsed schema information
        
    Returns:
        List of matched anchors with confidence scores
    """
    anchors = []
    concept_set = {c.lower() for c in concepts}
    
    for table_name, table_node in schema_info.tables.items():
        table_lower = table_name.lower()
        
        # Exact match
        if table_lower in concept_set:
            anchors.append(
                {
                    "id": table_name,
                    "type": "table",
                    "confidence": 1.0,
                    "hit_type": "name_match",
                    "table": table_name,
                }
            )
        # Partial match (table name contains concept or vice versa)
        else:
            for concept in concepts:
                concept_lower = concept.lower()
                if concept_lower in table_lower or table_lower in concept_lower:
                    anchors.append(
                        {
                            "id": table_name,
                            "type": "table",
                            "confidence": 0.7,
                            "hit_type": "partial_match",
                            "table": table_name,
                            "matched_concept": concept,
                        }
                    )
                    break
    
    return anchors


def match_potential_columns(
    potential_columns: List[str], schema_info: SchemaInfo
) -> List[Dict[str, Any]]:
    """
    Match potential column names against actual schema columns.
    
    Args:
        potential_columns: List of potential column names
        schema_info: Parsed schema information
        
    Returns:
        List of matched column anchors
    """
    anchors = []
    potential_set = {c.lower() for c in potential_columns}
    
    for table_name, table_node in schema_info.tables.items():
        for col_name in table_node.columns:
            col_lower = col_name.lower()
            if col_lower in potential_set:
                anchors.append(
                    {
                        "id": f"{table_name}.{col_name}",
                        "type": "column",
                        "confidence": 0.8,
                        "hit_type": "column_name_match",
                        "table": table_name,
                        "column": col_name,
                    }
                )
    
    return anchors


def infer_domain(concepts: List[str], operations: List[str]) -> str:
    """
    Infer domain from concepts and operations.
    
    Args:
        concepts: Extracted concepts
        operations: Extracted operations
        
    Returns:
        Inferred domain string
    """
    concepts_lower = [c.lower() for c in concepts]
    
    # Finance/Transaction domain
    if any(
        term in concepts_lower
        for term in ["transaction", "payment", "currency", "customer", "price", "amount"]
    ):
        return "finance_transaction"
    
    # Time-series domain
    if any(term in concepts_lower for term in ["date", "time", "year", "month", "consumption"]):
        return "time_series"
    
    # Product/Catalog domain
    if any(term in concepts_lower for term in ["product", "item", "catalog", "description"]):
        return "product_catalog"
    
    # Default
    return "general"


def build_graph_anchors(
    structured_entities: Dict[str, Any], schema_info: SchemaInfo
) -> List[Dict[str, Any]]:
    """
    Build graph anchors from structured entities and schema.
    Combines value matching, concept matching, and column matching.
    
    Args:
        structured_entities: Output from extract_structured_entities_llm
        schema_info: Parsed schema information
        
    Returns:
        List of unique graph anchors (deduplicated)
    """
    anchors = []
    seen_anchors = set()
    
    # Value matching (highest priority)
    value_anchors = match_values_against_examples(
        structured_entities.get("values", []), schema_info
    )
    for anchor in value_anchors:
        anchor_id = anchor["id"]
        if anchor_id not in seen_anchors:
            seen_anchors.add(anchor_id)
            anchors.append(anchor)
    
    # Concept matching
    concept_anchors = match_concepts_against_tables(
        structured_entities.get("concepts", []), schema_info
    )
    for anchor in concept_anchors:
        anchor_id = anchor["id"]
        if anchor_id not in seen_anchors:
            seen_anchors.add(anchor_id)
            anchors.append(anchor)
    
    # Column name matching
    column_anchors = match_potential_columns(
        structured_entities.get("potential_columns", []), schema_info
    )
    for anchor in column_anchors:
        anchor_id = anchor["id"]
        if anchor_id not in seen_anchors:
            seen_anchors.add(anchor_id)
            anchors.append(anchor)
    
    # Sort by confidence (descending)
    anchors.sort(key=lambda x: x["confidence"], reverse=True)
    
    return anchors
