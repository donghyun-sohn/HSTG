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
from typing import List

import numpy as np
from tqdm import tqdm


def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_embeddings_batch(
    texts: List[str], model: str = "llama3.2", batch_size: int = 10
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
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if not _ollama_available():
        print("Warning: Ollama server not reachable. Using random vectors for testing.")
        return [np.random.rand(1536).astype(np.float32) for _ in range(len(texts))]

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
                        batch_embs.append(np.array(embedding, dtype=np.float32))
                    else:
                        # Fallback to random if no embedding returned
                        batch_embs.append(np.random.rand(1536).astype(np.float32))
                        
            except urllib.error.URLError as e:
                print(f"Error embedding text in batch {i}: {e}")
                # Fill with random vector on error
                batch_embs.append(np.random.rand(1536).astype(np.float32))
            except Exception as e:
                print(f"Unexpected error in batch {i}: {e}")
                batch_embs.append(np.random.rand(1536).astype(np.float32))
        
        embeddings.extend(batch_embs)
            
    return embeddings
