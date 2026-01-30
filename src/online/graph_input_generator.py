#!/usr/bin/env python3
"""
Generate structured graph traversal input JSON from question and schema.
Integrates entity extraction, schema linking, and anchor scoring.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
from src.common.types import SchemaInfo
from src.online.entity_extractor import extract_structured_entities_llm
from src.online.schema_linking import build_graph_anchors, infer_domain


def generate_graph_input(
    question: str,
    schema_info: SchemaInfo,
    question_id: Optional[int] = None,
    evidence: str = "",
    max_hops: int = 2,
) -> Dict[str, Any]:
    """
    Generate structured graph traversal input JSON.
    
    Args:
        question: Natural language question
        schema_info: Parsed schema information
        question_id: Optional question ID
        evidence: Optional evidence/formula hint from BIRD
        max_hops: Maximum graph traversal hops
        
    Returns:
        Structured JSON for graph traversal
    """
    # Step 1: Extract structured entities
    table_names = list(schema_info.tables.keys())
    structured_entities = extract_structured_entities_llm(
        query=question,
        evidence=evidence,
        table_names=table_names,
    )
    
    # Step 2: Build graph anchors
    graph_anchors = build_graph_anchors(structured_entities, schema_info)
    
    # Step 3: Infer domain and intent
    domain = infer_domain(
        structured_entities.get("concepts", []),
        structured_entities.get("operations", []),
    )
    
    # Infer intent from operations
    operations = structured_entities.get("operations", [])
    intent = "general_query"
    if any(op.upper() in ["RATIO", "DIVIDE", "/"] for op in operations):
        intent = "ratio_calculation"
    elif any(op.upper() in ["COUNT", "SUM", "AVG"] for op in operations):
        intent = "aggregation"
    elif any(op.upper() in ["MAX", "MIN"] for op in operations):
        intent = "extremum_query"
    
    # Step 4: Build semantic gravity (domain + key concepts)
    # Note: This string representation can later be converted to an embedding vector
    # for similarity computation with Super Node centroids.
    # Example: Use embedder.get_embeddings_batch([semantic_gravity]) to get vector representation
    concepts_str = "_".join(structured_entities.get("concepts", [])[:3])
    semantic_gravity = f"{domain}_{concepts_str}" if concepts_str else domain
    
    # Step 5: Construct final JSON
    result = {
        "question_metadata": {
            "id": question_id,
            "db_id": schema_info.db_id,
        },
        "graph_anchors": graph_anchors,
        "traversal_hints": {
            "intent": intent,
            "operators": [op.upper() for op in operations],
            "semantic_gravity": semantic_gravity,
            "max_hops": max_hops,
        },
    }
    
    # Add evidence SQL stub if available
    if evidence:
        result["evidence_sql_stub"] = evidence
    
    return result


def get_semantic_gravity_embedding(
    semantic_gravity: str,
    embedder_func=None,
) -> Optional[np.ndarray]:
    """
    Convert semantic_gravity string to embedding vector for similarity computation.
    
    This function can be used to compute cosine similarity between semantic_gravity
    and Super Node centroids for graph routing.
    
    Args:
        semantic_gravity: Semantic gravity string (e.g., "finance_transaction_customers_currency")
        embedder_func: Optional embedding function. If None, returns None.
                      Should accept List[str] and return List[np.ndarray]
                      Example: from src.offline.embedder import get_embeddings_batch
    
    Returns:
        Embedding vector as numpy array, or None if embedder_func not provided
    
    Example:
        from src.offline.embedder import get_embeddings_batch
        
        graph_input = generate_graph_input(...)
        semantic_gravity = graph_input["traversal_hints"]["semantic_gravity"]
        embedding = get_semantic_gravity_embedding(
            semantic_gravity,
            embedder_func=lambda texts: get_embeddings_batch(texts)
        )
        
        # Compute similarity with Super Node centroid
        similarity = np.dot(embedding, super_node_centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(super_node_centroid)
        )
    """
    if embedder_func is None:
        return None
    
    try:
        embeddings = embedder_func([semantic_gravity])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
    except Exception:
        pass
    
    return None
