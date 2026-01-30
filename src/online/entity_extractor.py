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
from typing import List, Dict, Any


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


def extract_entities_llm(
    query: str, 
    model: str = "llama3.2",
    table_names: List[str] | None = None
) -> List[str]:
    """
    Extract entities using LLM (Ollama).
    
    Args:
        query: Natural language question
        model: Ollama model name
        table_names: Optional list of table names from the database schema.
                     If provided, the LLM will be instructed to prioritize
                     keywords that match or relate to these table names.
    
    Returns:
        List of extracted keywords
    """
    # Build prompt with table context if available
    if table_names and len(table_names) > 0:
        table_list = ", ".join(table_names)
        prompt = (
            "Task: Extract ALL possible database search keywords from the query.\n"
            "Instructions:\n"
            "1. Map verbs to business concepts (e.g., 'bought' -> 'transaction', 'purchase').\n"
            "2. Extract ALL specific nouns, adjectives, and domain terms.\n"
            "3. CRITICAL: The database contains these tables: " + table_list + "\n"
            "   - For EACH table name, extract keywords that could relate to it (exact matches, variations, synonyms, related concepts).\n"
            "   - Include singular/plural forms (customer/customers, product/products).\n"
            "   - Include abbreviations and full forms (e.g., 'SME' and 'small medium enterprise').\n"
            "   - Extract column-like terms (e.g., 'ID', 'Date', 'Amount', 'Price', 'Currency').\n"
            "   - Extract domain-specific terms from the query that might map to table concepts.\n"
            "   - Be EXTREMELY comprehensive: include ANY keyword that MIGHT be relevant, even if you're only 20% sure.\n"
            "   - Extract as many keywords as possible - aim for 10-20 keywords minimum.\n"
            "4. OUTPUT FORMAT: ONLY comma-separated words. NO intro text. NO explanations.\n\n"
            f"Query: {query}\n"
            "Keywords:"
        )
    else:
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
    # Clean up text: remove newlines and extra whitespace
    text = " ".join(text.split())
    # Extract keywords: split by comma, clean each keyword
    keywords = []
    for t in text.split(","):
        # Remove newlines, extra spaces, and clean
        kw = " ".join(t.split()).strip()
        if kw:
            keywords.append(kw)
    
    # Deduplicate while preserving order (case-insensitive)
    seen = set()
    deduped = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            deduped.append(kw)
    return deduped


def extract_structured_entities_llm(
    query: str,
    evidence: str = "",
    model: str = "llama3.2",
    table_names: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Extract structured entities from query and evidence using LLM.
    Returns a structured JSON with concepts, values, operations, and potential columns.
    
    Args:
        query: Natural language question
        evidence: Optional evidence/formula hint from BIRD dataset
        model: Ollama model name
        table_names: Optional list of table names from the database schema
    
    Returns:
        Dict with keys: concepts, values, operations, potential_columns
    """
    # Build prompt for structured extraction
    table_context = ""
    if table_names and len(table_names) > 0:
        table_list = ", ".join(table_names)
        table_context = f"\n\nThe database contains these tables: {table_list}"
    
    evidence_context = ""
    if evidence:
        evidence_context = f"\n\nEvidence/Formula: {evidence}"
    
    prompt = (
        "Task: Extract structured information from the query and evidence.\n"
        "Instructions:\n"
        "1. Extract CONCEPTS: domain entities, table-like terms, business concepts (e.g., 'customers', 'transaction', 'product').\n"
        "2. Extract VALUES: specific literal values mentioned (e.g., 'EUR', 'CZK', '2012', 'SME').\n"
        "3. Extract OPERATIONS: SQL operations implied (e.g., 'COUNT', 'SUM', 'AVG', 'DIVIDE', 'RATIO', 'MAX', 'MIN').\n"
        "4. Extract POTENTIAL_COLUMNS: column names that might be referenced (e.g., 'Currency', 'Date', 'Amount', 'Price').\n"
        f"{table_context}"
        f"{evidence_context}"
        "\n\nOUTPUT FORMAT: Return ONLY a valid JSON object with these exact keys:\n"
        '{\n'
        '  "concepts": ["list", "of", "concepts"],\n'
        '  "values": ["list", "of", "values"],\n'
        '  "operations": ["list", "of", "operations"],\n'
        '  "potential_columns": ["list", "of", "columns"]\n'
        '}\n'
        "NO intro text. NO explanations. ONLY the JSON object.\n\n"
        f"Query: {query}\n"
        "Extracted JSON:"
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
    # Clean up text: remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    
    # Try to extract JSON from the response
    try:
        # Find JSON object in the response
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return empty structure
        result = {
            "concepts": [],
            "values": [],
            "operations": [],
            "potential_columns": [],
        }
    
    # Ensure all required keys exist
    return {
        "concepts": result.get("concepts", []),
        "values": result.get("values", []),
        "operations": result.get("operations", []),
        "potential_columns": result.get("potential_columns", []),
    }


def extract_entities(
    query: str, 
    mode: str = "auto",
    table_names: List[str] | None = None
) -> List[str]:
    """
    Extract entities/keywords from a natural language query.
    
    Args:
        query: Natural language question
        mode: "auto" (try LLM, fallback to rule), "llm", or "rule"
        table_names: Optional list of table names from the database schema.
                     Used to guide LLM extraction to prioritize relevant keywords.
        
    Returns:
        List of extracted keywords
    """
    if mode == "llm":
        return extract_entities_llm(query, table_names=table_names)
    if mode == "rule":
        return extract_entities_rule_based(query)
    if mode == "auto":
        if _ollama_available():
            return extract_entities_llm(query, table_names=table_names)
        return extract_entities_rule_based(query)
    raise ValueError("mode must be one of: auto, llm, rule")
