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
    │   ├── schema_preprocess/       # Schema preprocessing package
    │   │   ├── __init__.py          # Package exports
    │   │   ├── parser.py            # DDL parsing (CREATE TABLE → TableNode)
    │   │   ├── preprocessor.py      # Semantic names + structural types
    │   │   └── serializer.py        # Table serialization for embedding
    │   ├── embedder.py              # Vector generation (Ollama)
    │   ├── graph_builder.py         # Hybrid graph from PreprocessedSchema
    │   ├── clustering.py            # Louvain community detection
    │   └── find_missing_fk.py       # Analyze implicit FK relationships
    │
    ├── online/                      # [Inference] Search & path finding
    │   ├── entity_extractor.py      # Keyword extraction (Ollama)
    │   ├── schema_linking.py        # Entity-to-schema matching
    │   ├── graph_input_generator.py # JSON input for graph traversal
    │   └── query_processor.py       # Process questions and generate SQL
    │
    └── test/                        # Test & demo scripts
        ├── test_preprocessing.py    # Schema preprocessing demo
        ├── test_keywords.py         # Keyword extraction test
        ├── test_three_questions.py  # Graph input JSON demo
        ├── inspect_graph.py         # Inspect generated graphs
        └── compare_strategies.py    # Inspect graph and clusters
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

### Step 0.1: Schema Preprocessing (Optional Demo)

Transform parsed schema into the format optimized for graph construction and similarity search:

- **Semantic Names**: Fully Qualified Names (FQN) with PK/FK annotations and example data
- **Structural Types**: Column data types for hard constraints

Run the preprocessing demo:

```bash
python -m src.test.test_preprocessing
```

This reads `mini_dev-main/finetuning/inference/mini_dev_prompt.jsonl`, parses schemas, and prints:
- Semantic names per table (e.g. `yearmonth.CustomerID (PK, references customers.CustomerID) (e.g. 39, 63)`)
- Structural types (e.g. `["integer", "text", "real"]`)
- FK mappings for hard edge creation
- Type-compatible pairs for soft edge candidates

**Code files used:**

| File | Role |
|------|------|
| `src/common/loader.py` | `load_unique_schemas()` - Load unique DB schemas from JSONL |
| `src/offline/schema_preprocess/parser.py` | `parse_bird_schema()` - Parse DDL into `TableNode` (columns, PK, FK, examples) |
| `src/offline/schema_preprocess/preprocessor.py` | `preprocess_schema()` - Transform to semantic names + structural types |
| `src/offline/schema_preprocess/serializer.py` | `serialize_schema()` - Table serialization for embedding |
| `src/common/types.py` | Data classes: `TableNode`, `SchemaInfo`, `PreprocessedTable`, `PreprocessedColumn` |

**Data flow:**

```
JSONL ──→ loader.py ──→ parser.py ──→ preprocessor.py
              │             │               │
         {db_id: ...}  SchemaInfo    PreprocessedSchema
                       (TableNode)   (semantic_names, structural_types)
```

**Preprocessed output: data types and structure**

The result is a `PreprocessedSchema` object per database. It is passed to downstream steps (graph building, embedding) as follows:

| Type | Structure | Purpose |
|------|-----------|---------|
| `PreprocessedSchema` | `db_id: str`, `tables: Dict[str, PreprocessedTable]`, `fk_mapping: Dict[Tuple[str,str], Tuple[str,str]]` | Root object for one database |
| `PreprocessedTable` | `table_name`, `context`, `semantic_names: List[str]`, `structural_types: List[str]`, `columns: List[PreprocessedColumn]` | One table with semantic names and types |
| `PreprocessedColumn` | `table_name`, `column_name`, `data_type`, `is_primary_key`, `foreign_key_ref`, `fk_points_to_pk`, `examples` | One column; `fk_points_to_pk` True if FK references a PK (for future graph weight boost) |

**Example (single table):**

```
PreprocessedSchema(db_id="debit_card_specializing", tables={...}, fk_mapping={...})
  └── tables["yearmonth"]: PreprocessedTable
        ├── semantic_names: ["yearmonth.CustomerID (PK, references customers.CustomerID) (e.g. 39, 63)", ...]
        ├── structural_types: ["integer", "text", "real"]
        └── columns: [PreprocessedColumn(...), ...]
```

**Downstream usage:**

- `semantic_names` → Embedding input (vectorization for similarity search)
- `structural_types` → Type constraint filter (same-type columns only for soft edges)
- `fk_mapping` → Hard edge creation `{(src_table, src_col): (ref_table, ref_col)}`
- `serialize_preprocessed_table()` → Produces embedding-ready string from `PreprocessedTable`

### Step 1: Offline Graph Construction

Build graphs and clusters for all databases (one-time preprocessing):

```bash
python -m src.test.test_bird_mini
```

This runs the full pipeline: Step 0 (preprocess) + Step 1 (graph) + Step 2 (clustering + save).

**Pipeline per database:**
1. Load schema from JSONL → parse → preprocess (Step 0)
2. Build hybrid graph from `PreprocessedSchema`
3. Run Louvain clustering
4. Save to `data/processed/db_graphs/{db_id}.pkl` and `clusters/{db_id}.json`

**Graph construction details (4 sub-steps):**

| Step | Description |
|------|-------------|
| **1. Node init & vectorization** | Each table = one node. Embed each column's `semantic_name` (Ollama). Node stores `column_vectors` (per column) and `vector` (mean for clustering). |
| **2. Hard Edge** | From `fk_mapping`. Connect `(src_table, src_col)` → `(ref_table, ref_col)`. Weight: `W_hard = 1.0 + (Sim_vec × 0.5)`. These are the primary JOIN paths. |
| **3. Soft Edge** | No explicit FK. Filter: `structural_types` must match. Weight: `W_soft = (Sim_name × 0.6) + (Sim_vec × 0.4)`. Prune if weight ≤ 0.6. |
| **4. Edge metadata** | Each edge stores `join_keys: [{source_col, target_col, type}]` for later `JOIN ON src.col = tgt.col` generation. |

**Code files used:**

| File | Role |
|------|------|
| `src/offline/graph_builder.py` | `build_hybrid_graph(prep_schema)` – node init, hard/soft edges, join_keys |
| `src/offline/embedder.py` | `get_embeddings_batch()` – Ollama column embedding |
| `src/offline/clustering.py` | `detect_communities()` – Louvain community detection |

**Node attributes:**
- `vector`: Table-level vector (mean of column vectors) for clustering
- `column_vectors`: Dict of column name → embedding
- `columns`, `structural_types`

**Edge attributes:**
- `weight`, `type` ("hard" or "soft")
- `join_keys`: `[{source_col, target_col, type}]`

**TODO for Junho:** `compute_column_vector_similarity()` in `graph_builder.py` – implements vector similarity for `Sim_vec`. Currently a placeholder (returns 0.0).

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

### Other Test Commands

| Command | Description |
|---------|-------------|
| `python -m src.test.test_bird_mini` | **Main pipeline**: Step 0 + 1 + 2 (preprocess, graph, clustering, save) |
| `python -m src.test.test_preprocessing` | Schema preprocessing demo (Step 0.1) |
| `python -m src.test.test_keywords` | Keyword extraction test |
| `python -m src.test.test_three_questions` | Graph input JSON for questions 1471–1473 |
| `python -m src.test.inspect_graph --db debit_card_specializing` | Inspect generated graph and clusters |
| `python -m src.test.compare_strategies --db debit_card_specializing` | Inspect graph and clusters |
| `python -m src.offline.find_missing_fk` | Find implicit FK relationships in SQL |

## Pipeline Overview

| Phase | Module | Command | Output |
|-------|--------|---------|--------|
| **Step 0.1** Schema Preprocessing | `schema_preprocess/preprocessor.py` | `python -m src.test.test_preprocessing` | Semantic names + structural types (demo) |
| **Step 1** Offline Graph Build | `test_bird_mini.py` | `python -m src.test.test_bird_mini` | Graph + clusters saved to `data/processed/` |
| **Step 2** Online Query | `query_processor.py` | `python -m src.online.query_processor` | Keywords, schema linking, graph input JSON |

Schema preprocessing feeds directly into the graph builder. Graph uses column-level vectors, FK-based hard edges, and type-constrained soft edges with join key metadata.

## Notes

- `--mode auto` uses Ollama if reachable, otherwise falls back to rule-based.
- Set `OLLAMA_BASE_URL` if your Ollama server is not on `http://localhost:11434`.
- The offline step only needs to run once (or when schemas change).
- Online step requires pre-built graphs. Run `python -m src.test.test_bird_mini` first.

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
