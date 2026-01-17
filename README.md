# Hierarchical Semantic-Topological Graph (HSTG) 

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

## BIRD mini-dev (bird_mini) download

If you want the mini-dev dataset for quick tests, use the official repo:

Clone the dataset **into this project root** so the folder name is `mini_dev-main`:

```
git clone https://github.com/bird-bench/mini_dev.git mini_dev-main
```

You can also download a ZIP and unzip it as `mini_dev-main` in this folder.

Then you can run:

```
python /Users/donghyunsohn/Downloads/HSTG/main.py --mode rule --limit 5
```
