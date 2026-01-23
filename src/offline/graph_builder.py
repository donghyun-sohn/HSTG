#!/usr/bin/env python3
"""
Hybrid graph builder combining hard edges (FK) and soft edges (semantic similarity).
Supports multiple edge weighting strategies for comparison.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Dict, List

from src.common.types import TableNode


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / norm if norm > 0 else 0.0


def compute_name_overlap(cols1: List[str], cols2: List[str]) -> float:
    """
    Compute name-based similarity using Jaccard similarity.
    Heuristic: Boost score if ID/code columns overlap.
    
    Args:
        cols1: Column names from table 1
        cols2: Column names from table 2
        
    Returns:
        Similarity score between 0 and 1
    """
    s1, s2 = set(cols1), set(cols2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    
    score = intersection / union if union > 0 else 0.0
    
    # [Heuristic] Boost if ID/code columns overlap
    for col in (s1 & s2):
        if "id" in col.lower() or "code" in col.lower():
            score = min(1.0, score + 0.3)  # Boost
            
    return score


def build_hybrid_graph_binary(
    tables: Dict[str, TableNode], vectors: Dict[str, np.ndarray]
) -> nx.Graph:
    """
    Strategy 1: Binary Hard/Soft Edge (Original)
    
    - Hard edges (FK): weight = 1.0
    - Soft edges (no FK): weight = name_sim * 0.6 + vec_sim * 0.4 (if > 0.6)
    
    This treats all FK relationships equally regardless of semantic similarity.
    """
    G = nx.Graph()
    table_names = list(tables.keys())
    
    # 1. Add nodes with vectors
    for name in table_names:
        G.add_node(name, vector=vectors[name])
        
    # 2. Add edges
    for i in range(len(table_names)):
        for j in range(i + 1, len(table_names)):
            t1_name, t2_name = table_names[i], table_names[j]
            t1, t2 = tables[t1_name], tables[t2_name]
            
            # Check for FK relationship (case-insensitive)
            t1_fks_lower = [fk.lower() for fk in t1.explicit_fks]
            t2_fks_lower = [fk.lower() for fk in t2.explicit_fks]
            is_fk = (t2_name.lower() in t1_fks_lower) or (t1_name.lower() in t2_fks_lower)
            
            if is_fk:
                # Hard Edge: Fixed weight 1.0
                G.add_edge(t1_name, t2_name, weight=1.0, type="hard")
            else:
                # Soft Edge: Name + Semantic similarity
                vec_sim = compute_cosine_similarity(vectors[t1_name], vectors[t2_name])
                name_sim = compute_name_overlap(t1.columns, t2.columns)
                
                soft_weight = (name_sim * 0.6) + (vec_sim * 0.4)
                
                # Threshold: Only connect if similarity > 0.6
                if soft_weight > 0.6:
                    G.add_edge(t1_name, t2_name, weight=soft_weight, type="soft")
                    
    return G


def build_hybrid_graph_reinforced(
    tables: Dict[str, TableNode], vectors: Dict[str, np.ndarray]
) -> nx.Graph:
    """
    Strategy 2: Reinforced Hard Edge (Proposed)
    
    - Hard edges (FK): weight = 1.0 + (vec_sim * 0.5)  [Range: 1.0 ~ 1.5]
    - Soft edges (no FK): weight = name_sim * 0.6 + vec_sim * 0.4 (if > 0.6)
    
    This reinforces FK relationships with semantic similarity, creating a hierarchy:
    - FK + High semantic similarity → Very strong connection (1.0 ~ 1.5)
    - FK + Low semantic similarity → Moderate connection (1.0 ~ 1.25)
    - No FK but high similarity → Inferred connection (0.6 ~ 1.0)
    
    Benefits:
    - Distinguishes tight business relationships (Order-Item) from loose logging links
    - Maintains structural priority (FK > inferred) while adding flexibility
    - Improves clustering quality by grouping semantically related tables
    """
    G = nx.Graph()
    table_names = list(tables.keys())
    
    # 1. Add nodes with vectors
    for name in table_names:
        G.add_node(name, vector=vectors[name])
        
    # 2. Add edges
    for i in range(len(table_names)):
        for j in range(i + 1, len(table_names)):
            t1_name, t2_name = table_names[i], table_names[j]
            t1, t2 = tables[t1_name], tables[t2_name]
            
            # Pre-compute semantic similarity (used in both cases)
            vec_sim = compute_cosine_similarity(vectors[t1_name], vectors[t2_name])
            
            # Check for FK relationship (case-insensitive)
            t1_fks_lower = [fk.lower() for fk in t1.explicit_fks]
            t2_fks_lower = [fk.lower() for fk in t2.explicit_fks]
            is_fk = (t2_name.lower() in t1_fks_lower) or (t1_name.lower() in t2_fks_lower)
            
            weight = 0.0
            edge_type = "none"
            
            if is_fk:
                # Reinforced Hard Edge: Base 1.0 + Semantic bonus
                # Weight range: 1.0 (low semantic) ~ 1.5 (high semantic)
                weight = 1.0 + (vec_sim * 0.5)
                edge_type = "hard_reinforced"
            else:
                # Soft Edge: Name + Semantic similarity
                name_sim = compute_name_overlap(t1.columns, t2.columns)
                soft_weight = (name_sim * 0.6) + (vec_sim * 0.4)
                
                # Threshold: Only connect if similarity > 0.6
                if soft_weight > 0.6:
                    weight = soft_weight
                    edge_type = "soft_inferred"
            
            # Add edge if weight > 0
            if weight > 0:
                G.add_edge(t1_name, t2_name, weight=weight, type=edge_type)
                    
    return G


def build_hybrid_graph(
    tables: Dict[str, TableNode],
    vectors: Dict[str, np.ndarray],
    strategy: str = "reinforced",
) -> nx.Graph:
    """
    Build hybrid graph with specified edge weighting strategy.
    
    Args:
        tables: Dict mapping table_name -> TableNode
        vectors: Dict mapping table_name -> embedding vector
        strategy: "binary" or "reinforced" (default: "reinforced")
        
    Returns:
        NetworkX graph with nodes and weighted edges
    """
    if strategy == "binary":
        return build_hybrid_graph_binary(tables, vectors)
    elif strategy == "reinforced":
        return build_hybrid_graph_reinforced(tables, vectors)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'binary' or 'reinforced'")
