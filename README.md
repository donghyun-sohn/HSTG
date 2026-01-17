# HSTG Entity Extraction Demo

Quick demo for hierarchical schema linking: extract entities/keywords from a
natural-language query, with optional local LLM (Ollama).

## Run

Rule-based (fast, no dependencies):

```
python /Users/donghyunsohn/Downloads/HSTG/input.py \
  --query "Which customer bought the most expensive product?" \
  --mode rule
```

LLM-based (local Ollama):

1. Start Ollama and pull a model:
```
ollama run llama3.2
```

2. Run with LLM mode:
```
python /Users/donghyunsohn/Downloads/HSTG/input.py \
  --query "Which customer bought the most expensive product?" \
  --mode llm
```

## Notes

- `--mode auto` uses Ollama if reachable, otherwise falls back to rule-based.
- Set `OLLAMA_BASE_URL` if your Ollama server is not on `http://localhost:11434`.
