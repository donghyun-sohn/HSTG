#!/usr/bin/env python3
"""
Entity extraction for hierarchical schema linking.
- Fast, local, rule-based extraction for immediate demos.
- Optional LLM-based extraction via Ollama.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import List


STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "in", "on", "at", "by", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "which", "who", "whom", "what", "where", "when", "why", "how",
    "this", "that", "these", "those", "it", "its", "their", "there",
    "most", "least", "more", "less", "many", "much",
}

PHRASE_HINTS = [
    (re.compile(r"\bmost expensive\b", re.I), ["price", "cost"]),
    (re.compile(r"\bhighest price\b", re.I), ["price", "cost"]),
    (re.compile(r"\blowest price\b", re.I), ["price", "cost"]),
    (re.compile(r"\bcount of\b", re.I), ["count"]),
]


def _simple_singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def extract_entities_rule_based(query: str) -> List[str]:
    lowered = query.lower()
    results: List[str] = []

    # Phrase hints first (captures multi-word semantics like "most expensive")
    for pattern, concepts in PHRASE_HINTS:
        if pattern.search(lowered):
            results.extend(concepts)

    # Tokenization
    tokens = re.findall(r"[a-zA-Z0-9_]+", lowered)
    for token in tokens:
        if token in STOPWORDS:
            continue
        token = _simple_singularize(token)
        results.append(token)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for item in results:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _ollama_available() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def extract_entities_llm(query: str, model: str = "llama3.2") -> List[str]:
    prompt = (
        "Task: Extract database search keywords from the query.\n"
        "Instructions:\n"
        "1. Map verbs to business concepts (e.g., 'bought' -> 'transaction').\n"
        "2. Extract specific nouns (e.g., 'customer', 'product').\n"
        "3. OUTPUT FORMAT: ONLY comma-separated words. NO intro text. NO explanations.\n\n"
        f"Query: {query}\n"
        "Keywords:"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama server is not reachable") from exc

    text = data.get("message", {}).get("content", "").strip()
    return [t.strip() for t in text.split(",") if t.strip()]


def extract_entities(query: str, mode: str = "auto") -> List[str]:
    """
    Extract entities/keywords from a natural language query.
    
    Args:
        query: Natural language question
        mode: "auto" (try LLM, fallback to rule), "llm", or "rule"
        
    Returns:
        List of extracted keywords
    """
    if mode == "llm":
        return extract_entities_llm(query)
    if mode == "rule":
        return extract_entities_rule_based(query)
    if mode == "auto":
        if _ollama_available():
            return extract_entities_llm(query)
        return extract_entities_rule_based(query)
    raise ValueError("mode must be one of: auto, llm, rule")
