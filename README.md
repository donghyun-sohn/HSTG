# Hierarchical Semantic-Topological Graph (HSTG)

Hierarchical schema linking for BIRD Benchmark using semantic clustering and graph-based routing.

## Project Structure

```
project_root/
├── mini_dev-main/                   # [User Provided] Unzipped folder
│   └── finetuning/inference/mini_dev_prompt.jsonl
├── data/
│   └── processed/                   # [Offline Output] Graph storage
│       ├── db_graphs/               # Store graphs per db_id (.pkl)
│       └── clusters/                # Store cluster info per db_id (.json)
│
└── src/
    ├── common/
    │   ├── loader.py                # Single-pass JSONL loader
    │   └── types.py                 # Schema, Table data classes
    │
    ├── offline/                     # [Preprocessing] Graph & clustering
    │   └── build_graph.py           # Build graphs per DB
    │
    ├── online/                      # [Inference] Search & path finding
    │   ├── entity_extractor.py     # Keyword extraction
    │   └── query_processor.py      # Process questions and generate SQL
    │
    └── test/                         # Test scripts
        └── test_keywords.py         # Keyword extraction test
```

## Setup

### 1. Download BIRD mini-dev

Clone the dataset **into this project root** so the folder name is `mini_dev-main`:

```bash
git clone https://github.com/bird-bench/mini_dev.git mini_dev-main
```

Or download a ZIP and unzip it as `mini_dev-main` in this folder.

### 2. (Optional) Setup Ollama for LLM-based extraction

```bash
ollama run llama3.2
```

## Usage

### Step 1: Offline Graph Construction

Build graphs and clusters for all databases (one-time preprocessing):

```bash
python -m src.offline.build_graph
```

This will:
- Load unique schemas from `mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl`
- For each `db_id`:
  - Create vertices (table names)
  - Create table indicator vectors
  - Connect edges (PK-FK relationships + semantic relationships)
  - Run clustering (semantic similarity, community detection)
  - Create super cluster indicator vectors
- Save results to `data/processed/`

### Step 2: Online Query Processing

Process questions using pre-built graphs:

```bash
python -m src.online.query_processor --mode rule --limit 5
```

Options:
- `--data`: Path to JSONL file (default: `mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl`)
- `--limit`: Number of questions to process
- `--mode`: Entity extraction mode (`auto`, `llm`, or `rule`)

### Standalone Entity Extraction

For quick testing without full pipeline, use the module directly:

```python
from src.online.entity_extractor import extract_entities
keywords = extract_entities("Which customer bought the most expensive product?", mode="rule")
print(keywords)
```

## Notes

- `--mode auto` uses Ollama if reachable, otherwise falls back to rule-based.
- Set `OLLAMA_BASE_URL` if your Ollama server is not on `http://localhost:11434`.
- The offline step only needs to run once (or when schemas change).
- Online step requires pre-built graphs from Step 1.

## Workflow

1. **OFFLINE** (One-time):
   - Creates Vertex with each table names
   - Create Table Indicator Vector
   - Connects edges with weights (PK_FK relationship + semantic relationship)
   - Clustering (semantic similarity, community detection)
   - Creates super cluster's indicator vector
   - Check edges for each super clusters

2. **ONLINE** (Per query):
   - Input: Business Questions → OLLAMA → keyword extraction
   - Find which supercluster matches for each keyword
   - Within super clusters, find all table names, col names
   - Give the list of table names to the LLM to create SQL
