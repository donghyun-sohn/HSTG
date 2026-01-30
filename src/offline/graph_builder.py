#!/usr/bin/env python3
"""
Hybrid graph builder from preprocessed schema.

Builds graph with:
- Node: Table (each node has column-level vectors)
- Hard Edge: FK relationships (W_hard = 1.0 + Sim_vec * 0.5)
- Soft Edge: Type-compatible, semantically similar columns (W_soft = Sim_name*0.6 + Sim_vec*0.4)
- Edge Metadata: source_col, target_col for JOIN key mapping
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
import numpy as np

from src.offline.schema_preprocess import (
    PreprocessedSchema,
    PreprocessedTable,
    PreprocessedColumn,
    get_type_compatible_pairs,
)
from src.offline.embedder import get_embeddings_batch


# =============================================================================
# TODO for Junho: Vector similarity between two columns
# =============================================================================
# This function computes cosine similarity between two column embedding vectors.
# Used for:
# - Hard Edge: W_hard = 1.0 + (Sim_vec * 0.5) between FK-linked columns
# - Soft Edge: W_soft = (Sim_name * 0.6) + (Sim_vec * 0.4) between type-compatible columns
# =============================================================================


def compute_column_vector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute vector similarity between two column embeddings.

    TODO for Junho: Implement this function.

    Used for:
    - Hard Edge: W_hard = 1.0 + (Sim_vec * 0.5) between FK-linked columns
    - Soft Edge: W_soft = (Sim_name * 0.6) + (Sim_vec * 0.4) between type-compatible columns

    Expected: cosine similarity. Typical range [0, 1] or [-1, 1].
    Current: placeholder returning 0.0 (hard edges get weight 1.0, soft edges use name only).

    Args:
        vec1: Embedding vector for column 1
        vec2: Embedding vector for column 2

    Returns:
        Similarity score (e.g. cosine similarity in [0, 1])
    """
    # TODO for Junho: implement vector similarity between two column embeddings
    return 0.0


def _compute_column_name_similarity(col1: str, col2: str) -> float:
    """
    Jaccard-like similarity between two column names.
    Used for W_soft = (Sim_name * 0.6) + (Sim_vec * 0.4).
    """
    c1, c2 = col1.lower(), col2.lower()
    if c1 == c2:
        return 1.0
    s1, s2 = set(c1.split("_")), set(c2.split("_"))
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    base = inter / union if union > 0 else 0.0
    if ("id" in c1 or "code" in c1) and ("id" in c2 or "code" in c2):
        base = min(1.0, base + 0.3)
    return base


def _embed_columns(prep_schema: PreprocessedSchema, model: str = "llama3.2") -> Dict[Tuple[str, str], np.ndarray]:
    """Embed each column's semantic_name. Returns (table, col) -> vector."""
    texts: List[str] = []
    keys: List[Tuple[str, str]] = []
    for table_name, prep_table in prep_schema.tables.items():
        for col in prep_table.columns:
            texts.append(col.to_semantic_name())
            keys.append((table_name, col.column_name))
    vectors = get_embeddings_batch(texts, model=model)
    return dict(zip(keys, vectors))


def build_hybrid_graph(
    prep_schema: PreprocessedSchema,
    model: str = "llama3.2",
    soft_threshold: float = 0.6,
) -> nx.Graph:
    """
    Build hybrid graph from PreprocessedSchema.

    Step 1: Node init & vectorization
        - Each table is a node
        - Each column's semantic_name is embedded (Ollama)
        - Node stores column_vectors, vector (mean for clustering)

    Step 2: Hard Edge (FK)
        - From fk_mapping
        - W_hard = 1.0 + (Sim_vec * 0.5)
        - Edge attr: join_keys with source_col, target_col, type="hard"

    Step 3: Soft Edge (semantic inference)
        - Type-compatible column pairs only
        - W_soft = (Sim_name * 0.6) + (Sim_vec * 0.4)
        - Prune: weight <= soft_threshold removed

    Step 4: Edge metadata
        - join_keys: source_col, target_col for JOIN key mapping
    """
    G = nx.Graph()
    tables = prep_schema.tables
    fk_mapping = prep_schema.fk_mapping

    column_vectors = _embed_columns(prep_schema, model=model)

    for table_name, prep_table in tables.items():
        table_col_vecs = {}
        vecs_for_centroid = []
        for col in prep_table.columns:
            vec = column_vectors.get((table_name, col.column_name))
            table_col_vecs[col.column_name] = vec
            if vec is not None and len(vec) > 0:
                vecs_for_centroid.append(vec)
        default_dim = len(next(iter(column_vectors.values()), np.zeros(256))) if column_vectors else 256
        table_vector = np.mean(vecs_for_centroid, axis=0) if vecs_for_centroid else np.zeros(default_dim)
        G.add_node(
            table_name,
            vector=table_vector,
            column_vectors=table_col_vecs,
            columns=[c.column_name for c in prep_table.columns],
            structural_types=prep_table.structural_types,
        )

    for (src_table, src_col), (ref_table, ref_col) in fk_mapping.items():
        vec_src = column_vectors.get((src_table, src_col))
        vec_ref = column_vectors.get((ref_table, ref_col))
        sim_vec = compute_column_vector_similarity(
            vec_src if vec_src is not None else np.zeros(1),
            vec_ref if vec_ref is not None else np.zeros(1),
        )
        weight = 1.0 + (sim_vec * 0.5)

        if G.has_edge(src_table, ref_table):
            existing = G.edges[src_table, ref_table]
            if weight > existing.get("weight", 0):
                G.edges[src_table, ref_table]["weight"] = weight
                join_keys = G.edges[src_table, ref_table].get("join_keys", [])
                join_keys.append({"source_col": src_col, "target_col": ref_col, "type": "hard"})
        else:
            G.add_edge(
                src_table, ref_table,
                weight=weight, type="hard",
                join_keys=[{"source_col": src_col, "target_col": ref_col, "type": "hard"}],
            )

    compatible_pairs = get_type_compatible_pairs(prep_schema)
    table_pairs: Dict[Tuple[str, str], List[Tuple[str, str, str, str]]] = {}
    for col1_fqn, type1, col2_fqn, type2 in compatible_pairs:
        t1, c1 = col1_fqn.split(".", 1)
        t2, c2 = col2_fqn.split(".", 1)
        if t1 > t2:
            t1, t2, c1, c2 = t2, t1, c2, c1
        key = (t1, t2)
        if key not in table_pairs:
            table_pairs[key] = []
        table_pairs[key].append((c1, c2, col1_fqn, col2_fqn))

    for (t1, t2), pairs in table_pairs.items():
        if G.has_edge(t1, t2):
            continue
        best_weight = 0.0
        best_join = None
        for c1, c2, fqn1, fqn2 in pairs:
            vec1 = column_vectors.get((t1, c1))
            vec2 = column_vectors.get((t2, c2))
            sim_name = _compute_column_name_similarity(c1, c2)
            sim_vec = compute_column_vector_similarity(
                vec1 if vec1 is not None else np.zeros(1),
                vec2 if vec2 is not None else np.zeros(1),
            )
            w = (sim_name * 0.6) + (sim_vec * 0.4)
            if w > best_weight:
                best_weight = w
                best_join = (c1, c2)
        if best_weight > soft_threshold and best_join:
            c1, c2 = best_join
            G.add_edge(
                t1, t2,
                weight=best_weight, type="soft",
                join_keys=[{"source_col": c1, "target_col": c2, "type": "soft"}],
            )

    return G
