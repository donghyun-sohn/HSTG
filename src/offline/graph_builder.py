#!/usr/bin/env python3
"""
Hybrid graph builder combining hard edges (FK) and soft edges (semantic similarity).
Core novelty: Topological + Semantic relationships.
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


def build_hybrid_graph(
    tables: Dict[str, TableNode], vectors: Dict[str, np.ndarray]
) -> nx.Graph:
    """
    Build hybrid graph with hard edges (FK) and soft edges (semantic similarity).
    
    Args:
        tables: Dict mapping table_name -> TableNode
        vectors: Dict mapping table_name -> embedding vector
        
    Returns:
        NetworkX graph with nodes and weighted edges
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
            
            # A. Hard Edge (Explicit FK)
            # Case-insensitive matching for FK references
            t1_fks_lower = [fk.lower() for fk in t1.explicit_fks]
            t2_fks_lower = [fk.lower() for fk in t2.explicit_fks]
            is_fk = (t2_name.lower() in t1_fks_lower) or (t1_name.lower() in t2_fks_lower)
            
            if is_fk:
                G.add_edge(t1_name, t2_name, weight=1.0, type="hard")
            else:
                # B. Soft Edge (Semantic + Name similarity)
                vec_sim = compute_cosine_similarity(vectors[t1_name], vectors[t2_name])
                name_sim = compute_name_overlap(t1.columns, t2.columns)
                
                # Weight formula: 60% name similarity + 40% semantic similarity (adjustable)
                soft_weight = (name_sim * 0.6) + (vec_sim * 0.4)
                
                # Threshold: Only connect if similarity > 0.6 (avoid too many edges)
                if soft_weight > 0.6:
                    G.add_edge(t1_name, t2_name, weight=soft_weight, type="soft")
                    
    return G
