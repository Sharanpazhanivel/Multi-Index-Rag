# Multi-Index RAG with Learned & Bandit Router

Multi-index RAG system where a router chooses, **per query**, which retrievers and settings to use (vector, BM25, SQL, code index) to balance **answer quality**, **latency**, and **cost**.

This repo contains a full research-style pipeline:

- **Multiple retrievers** (BM25, dense/vector, SQL, code)
- **Router evolution**: rules → supervised transformer → bandit / RL
- **End-to-end logging & feedback** for learning from usage
- **Evaluation & ablations** across different routing strategies

---

## 1. High-level architecture

### 1.1 End-to-end: query → answer

```text
    [User query]
            |
            v
    +-------+-------+
    |    Router     |
    +-------+-------+
            |
    +-------+-------+-------+-------+-------+
    |       |       |       |       |       |
    v       v       v       v       v
  BM25   Vector   SQL   Vector   Code
  only   only   only  + BM25  + Vector
    |       |       |       |       |
    v       v       v       v       v
  [BM25] [Vector] [SQL] [BM25+Vector] [Code+Vector]
    \       |       |       |       /
     \      +-------+-------+------+
      \             |
       \            v
        \    [Re-rank (RRF)]
         \          |
          \         v
           \    [LLM answer]
            \________|
```

### 1.2 Router evolution (Phase 1 → 2 → 3)

```text
  Phase 1: Rules
  +------------------+
  | Keyword / regex  |----+
  | Centroid sim     |--+
  +------------------+  |
            |           v
            v    [Combined rules router]
            |           |
            |           v
            |    router_log.jsonl
            |           |
            |           v
            |    [Offline eval] --------> routing_labels.jsonl
            |                                    |
  Phase 2: Learned                              v
  +------------------+                  [Train transformer]
  | routing_labels   |------------------------->|
  +------------------+                          v
                                        [Checkpoint] --> LearnedRouter
                                                |
  Phase 3: Bandit / RL                          |
  +------------------+                          v
  | LinUCB           |----+              Epsilon-greedy (uses LearnedRouter)
  | Eps-greedy       |----+----> [BanditRouter]
  | REINFORCE        |----+              REINFORCE (uses checkpoint)
  | feedback_log     |----+
  +------------------+
```

### 1.3 Data & feedback flow

```text
  data/raw  -->  data/processed
       |
  data/labels --> queries.jsonl
       |                |
       |                v
       |         phase1_collect_labels
       |                |
       |                +---> router_log.jsonl
       |                |
       |                +---> (--offline-eval) --> routing_labels.jsonl
       |                                                |
       |                                                v
       |                                         phase2_train_router
       |                                                |
       |                                                v
       |                                         checkpoints/router
       |
       v
  router_log.jsonl  -->  feedback_log.jsonl
                                |
                                v
                         phase3_bandit_online (replay)
                                |
                                v
                         checkpoints/router (REINFORCE --save)
```

### 1.4 Single request path (pipeline)

```text
  [Query]
      |
      v
  Router type? ----+---- Rules  ----> [Keyword / Centroid]
      |            +---- Learned --> [Transformer, argmax]
      |            +---- Bandit ----> [LinUCB / Eps-greedy / REINFORCE]
      |
      v
  [RouterDecision: action_id, retriever_names]
      |
      v
  [Log decision + latency]
      |
      v
  [Retrieve from chosen index(es)]
      |
      v
  Multiple retrievers? ---- Yes ----> [RRF merge, top-k]
      |                                |
      No                              v
      |                          [Chunks]
      +---------------------------------> [Chunks]
                      |
                      v
                [LLM answer]
                      |
                      v
            (Optional) record_feedback --> feedback_log.jsonl
```

---

## 2. Project structure

```text
.
├── config/           # Settings, index definitions, router actions
├── data/
│   ├── raw/          # Input data (JSONL / txt)        (gitignored by default)
│   ├── processed/    # Built indexes & logs            (gitignored; .gitkeep only)
│   └── labels/       # Query sets, routing labels, eval sets
├── notebooks/        # Exploration & analysis
├── scripts/          # Phase entrypoints (0–4)
├── src/
│   ├── api/          # Optional FastAPI-style service
│   ├── evaluation/   # Retrieval, answer & system metrics, ablations
│   ├── logging/      # Router & feedback logging helpers
│   ├── rag/          # Core RAG pipeline, LLM wrapper, reranker
│   ├── retrievers/   # BM25, vector, SQL, code, index builder
│   ├── router/       # Rules, learned and bandit routers
│   └── schema/       # Shared types (Chunk, RetrievalResult, RouterDecision)
├── tests/            # Unit tests
├── run.py            # Simple CLI demo entrypoint
├── requirements.txt
└── pyproject.toml
```

---

## 3. Getting started

### 3.1 Prerequisites

- Python **3.10+**
- Git
- (Optional) OpenAI API key for answer generation

### 3.2 Installation

```bash
git clone https://github.com/Sharanpazhanivel/Multi-Index-Rag.git
cd Multi-Index-Rag

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3.3 Environment

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY=your_key_here
```

`.env` is **gitignored** and never pushed.

---

## 4. Running the demo (single query)

The easiest way to see the system end-to-end:

```bash
python run.py
```

The script will:

1. Build simple sample indexes (if `data/raw` is empty) using BM25, vector, SQL, and code retrievers.  
2. Route your query through the current router (rules / learned / bandit).  
3. Retrieve, re-rank, and (optionally) call the LLM to generate an answer.  
4. Ask you whether the answer was helpful and log feedback to `data/processed/feedback_log.jsonl`.

---

## 5. Working with the phases

### 5.1 Phase 0 – Setup & index build

- Common schema types: `Chunk`, `RetrievalResult`, `RouterDecision`.
- Pluggable retriever interface (`BaseRetriever`) with BM25 (`general`), vector (`technical`), SQL (`structured`), and code (`code`) retrievers.
- Baseline RAG pipeline for querying one or more retrievers.
- Index and action configuration in:
  - `config/indexes.yaml`
  - `config/router_actions.yaml` (e.g. BM25-only, vector-only, SQL-only, vector+BM25, code+vector).

Build indexes:

```bash
python scripts/phase0_build_indexes.py
```

This reads from `data/raw/*.jsonl` / `*.txt` and writes indexes to `data/processed/`.

### 5.2 Phase 1 – Rules-based router

- **Keyword router**: Regex-based rules to route SQL-like questions to the structured retriever, code questions to code+vector, etc.
- **Centroid router**: Embedding similarity to per-action centroids (built via `scripts/build_centroids.py`).
- **Combined router**: Keyword first; falls back to centroid / default for ambiguous queries.
- **Re-ranker**: Reciprocal rank fusion (RRF) across multiple retrievers in `src/rag/reranker.py`.
- **Logging**: Router decisions to `router_log.jsonl`, user feedback to `feedback_log.jsonl`.

Collect labels and logs:

```bash
python scripts/phase1_collect_labels.py \
  --queries data/labels/queries.jsonl \
  --offline-eval        # optional; creates routing_labels.jsonl
```

### 5.3 Phase 2 – Learned (transformer) router

- Small encoder (default **DistilBERT**) + classification head → action id.
- Checkpoint layout: `router_config.json`, `model.pt`, `tokenizer/`.
- Training uses `routing_labels.jsonl` from Phase 1.

Train:

```bash
python scripts/phase2_train_router.py \
  --labels data/labels/routing_labels.jsonl \
  --out-dir checkpoints/router
```

Load in code:

```python
from src.router.learned import load_router

router = load_router("checkpoints/router")
```

Then plug into `RAGPipeline(router=router)`.

### 5.4 Phase 3 – Bandit / RL

- **LinUCB** contextual bandit (query embedding + cost/latency prefs).
- **Epsilon-greedy** exploration on top of a base router (e.g. learned).
- **REINFORCE** policy gradient over the transformer router; supports online updates and replay from logs.
- **BanditRouter** wrapper (`strategy=linucb|epsilon_greedy|reinforce`) with:
  - `route(query)`
  - `update(query, decision, reward)` or `update_from_log_entry(...)`.
- **OPE helpers** in `src/router/bandit/ope.py`.

Run bandit training / replay:

```bash
python scripts/phase3_bandit_online.py \
  --strategy linucb \
  --mode replay \
  --feedback-log data/processed/feedback_log.jsonl
```

### 5.5 Phase 4 – Evaluation & ablations

- Eval set stored as JSONL with `query`, optional `query_id`, `reference_answer`, `gold_doc_ids`.
- Metrics:
  - Retrieval: Hit@k, MRR
  - Answer: EM, F1
  - System: avg latency, cost, number of chunks
- Ablations runner: `run_ablations()` and `format_table()` in `src/evaluation`.

Run evaluation:

```bash
python scripts/phase4_run_eval.py \
  --eval-set data/labels/eval_set.jsonl \
  --checkpoint checkpoints/router \
  --output results.txt \
  --format table
```

---

## 6. Status

Implemented:

- ✅ Index build for BM25 / vector / SQL / code
- ✅ Rules-based router + logging
- ✅ Supervised transformer router (training + inference)
- ✅ Bandit / RL router (LinUCB, epsilon-greedy, REINFORCE) with OPE
- ✅ Evaluation utilities and phase scripts
- ✅ Simple CLI (`run.py`) to try everything with a single query

If you use this project or extend it (e.g., new retrievers, reward functions, or routers), feel free to open an issue or PR. :)
