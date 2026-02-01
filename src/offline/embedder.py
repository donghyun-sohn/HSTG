#!/usr/bin/env python3
"""
Embedding generator for table serializations.
Converts serialized table text into vectors using Ollama (local LLM).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm


DEFAULT_EMBEDDING_DIM = 4096
_EMBEDDING_DIM_CACHE: Dict[Tuple[str, str], int] = {}


def _get_base_url() -> str:
    """Return Ollama base URL honoring env override."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _cache_embedding_dim(model: str, base_url: str, vector: np.ndarray) -> None:
    """Persist the embedding dimension inferred from a returned vector."""
    if isinstance(vector, np.ndarray) and vector.size:
        _EMBEDDING_DIM_CACHE[(model, base_url)] = int(vector.shape[0])


def _has_embedding_dim(model: str, base_url: str) -> bool:
    return (model, base_url) in _EMBEDDING_DIM_CACHE


def get_embedding_dimension(model: str = "qwen3-embedding:8b", default: int = DEFAULT_EMBEDDING_DIM) -> int:
    """Expose the currently known embedding dim for the given model/base URL."""
    base_url = _get_base_url()
    return _EMBEDDING_DIM_CACHE.get((model, base_url), default)


def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    base_url = _get_base_url()
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _random_vector(dim: int) -> np.ndarray:
    """Generate a deterministic-typed random vector for fallback paths."""
    return np.random.rand(dim).astype(np.float32)


def get_embeddings_batch(
    texts: List[str], model: str = "qwen3-embedding:8b", batch_size: int = 10
) -> List[np.ndarray]:
    """
    Generate embeddings for texts using Ollama (local LLM).
    
    Args:
        texts: List of serialized table texts
        model: Ollama model name (must support embeddings)
        batch_size: Number of texts to process per batch (smaller for local)
        
    Returns:
        List of numpy arrays (embeddings)
    """
    embeddings = []
    base_url = _get_base_url()
    default_dim = get_embedding_dimension(model)
    
    if not _ollama_available():
        print("Warning: Ollama server not reachable. Using random vectors for testing.")
        return [_random_vector(default_dim) for _ in range(len(texts))]

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Tables (Ollama)"):
        batch = texts[i : i + batch_size]
        batch_embs = []
        
        for text in batch:
            try:
                # Replace newlines with spaces
                clean_text = text.replace("\n", " ")
                
                # Ollama embeddings API
                payload = json.dumps({"model": model, "prompt": clean_text}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    embedding = data.get("embedding", [])
                    
                    if embedding:
                        vec = np.array(embedding, dtype=np.float32)
                        batch_embs.append(vec)
                        if not _has_embedding_dim(model, base_url):
                            _cache_embedding_dim(model, base_url, vec)
                            default_dim = vec.shape[0]
                    else:
                        # Fallback to random if no embedding returned
                        rand_vec = _random_vector(default_dim)
                        batch_embs.append(rand_vec)
                        if not _has_embedding_dim(model, base_url):
                            _cache_embedding_dim(model, base_url, rand_vec)
                        
            except urllib.error.URLError as e:
                print(f"Error embedding text in batch {i}: {e}")
                # Fill with random vector on error
                rand_vec = _random_vector(default_dim)
                batch_embs.append(rand_vec)
                if not _has_embedding_dim(model, base_url):
                    _cache_embedding_dim(model, base_url, rand_vec)
            except Exception as e:
                print(f"Unexpected error in batch {i}: {e}")
                rand_vec = _random_vector(default_dim)
                batch_embs.append(rand_vec)
                if not _has_embedding_dim(model, base_url):
                    _cache_embedding_dim(model, base_url, rand_vec)
        
        embeddings.extend(batch_embs)
            
    # Cache dimension if the run succeeded but the first batch was empty
    for vec in embeddings:
        if isinstance(vec, np.ndarray) and vec.size:
            _cache_embedding_dim(model, base_url, vec)
            break

    return embeddings
