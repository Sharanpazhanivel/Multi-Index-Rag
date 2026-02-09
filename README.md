# Multi-Index RAG with Learned Router

Multi-index RAG system where a router chooses, per query, which retrievers and settings to use (vector, BM25, SQL, code index) to maximize answer quality under latency/cost constraints.

---

## What’s ## How it works

### End-to-end: query to answer

```
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
        \    [Re-rank RRF]
         \          |
          \         v
           \   [Answer stub]
            \________|
```

### Router evolution (Phase 1 to 2 to 3)

```
  Phase 1: Rules
  +------------------+
  | Keyword / regex   |----+
  | Centroid similarity |--+
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
  | Epsilon-greedy   |----+----> [BanditRouter]
  | REINFORCE        |----+              REINFORCE (uses Checkpoint)
  | feedback_log     |----+
  +------------------+
```

### Data and feedback flow

```
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
                         phase3_bandit_online replay
                                |
                                v
                         checkpoints/router (REINFORCE --save)
```

### Single request path (pipeline)

```
  [Query]
      |
      v
  Router type? ----+---- Rules -----> [Keyword / Centroid]
      |            +---- Learned ---> [Transformer, argmax]
      |            +---- Bandit -----> [LinUCB / Epsilon-greedy / REINFORCE]
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
                [LLM stub]
                      |
                      v
                [Answer]
                      |
                      v
                (Optional) record_feedback --> feedback_log.jsonl
```

---

## What's done so far

**Phase 0 – Setup**  
- **Schema**: Common types (`Chunk`, `RetrievalResult`, `RouterDecision`) so every retriever and router speaks the same format.  
- **Retrievers**: Pluggable interface (`BaseRetriever`) and four stubs: BM25 (`general`), vector (`technical`), SQL (`structured`), code (`code`), plus a registry.  
- **Baseline RAG**: Query one or more retrievers, get back chunks with latency/cost.  
- **Config**: `config/indexes.yaml`, `config/router_actions.yaml` (five actions: BM25 only, vector only, SQL only, vector+BM25, code+vector).

**Phase 1 – Rules-based router**  
- **Keyword router**: Regex rules (e.g. SQL → structured retriever, code patterns → code+vector).  
- **Centroid router**: Optional embedding similarity to per-action centroids (from `data/processed/centroids.json`); build centroids with `scripts/build_centroids.py`.  
- **Combined router**: Keyword first; when no SQL/code match, fall back to centroid (or default).  
- **Re-ranker**: Merge results from multiple retrievers with RRF (reciprocal rank fusion) in `src/rag/reranker.py`.  
- **Pipeline**: `RAGPipeline(router=...)` runs route → log decision + latency → retrieve → re-rank when multi-index.  
- **Logging**: Router decisions to `router_log.jsonl`, user feedback to `feedback_log.jsonl` (for Phase 2 labels and Phase 3 bandit).  
- **Scripts**: `phase1_collect_labels.py` (run router + log; optional `--offline-eval` to try all actions and write `routing_labels.jsonl` with best action per query).

**Phase 2 – Supervised transformer router**  
- **Model**: Small encoder (default DistilBERT) + classification head → action id; checkpoint = `router_config.json`, `model.pt`, `tokenizer/`.  
- **Training**: Load `routing_labels.jsonl`, cross-entropy training, validation top-1 accuracy, save checkpoint.  
- **Inference**: `load_router(checkpoint_dir)` returns a `LearnedRouter` that can replace the rules router in the pipeline.  
- **Script**: `phase2_train_router.py` (uses labels from Phase 1 `--offline-eval`).

**Phase 3 – Bandit / RL**  
- **LinUCB**: Contextual bandit (context = query embedding + cost/latency prefs); select by UCB, update with (context, action, reward).  
- **Epsilon-greedy**: With prob ε random action, else use base router (e.g. learned).  
- **REINFORCE**: Policy gradient over the transformer router; sample action from softmax, update with reward × grad(log π). Supports online (update after route) and replay (update from `feedback_log.jsonl` via re-forward for log_prob).  
- **BanditRouter**: Single `BaseRouter` with `strategy=linucb|epsilon_greedy|reinforce`; `route(query)` and `update(query, decision, reward)` or `update_from_log_entry(query, action_id, reward)`.  
- **OPE**: `mean_reward_from_log`, `evaluate_policy_on_log` in `src/router/bandit/ope.py`.  
- **Script**: `phase3_bandit_online.py --strategy linucb|epsilon_greedy|reinforce --mode online|replay` (replay updates from `feedback_log.jsonl`; optional `--save` for REINFORCE).

**Phase 4 – Evaluation and ablations**  
- **Eval set**: JSONL with `query`, optional `query_id`, `reference_answer`, `gold_doc_ids`; `load_eval_set()` in `src/evaluation/eval_set.py`.  
- **Metrics**: Retrieval (Hit@5, MRR), answer (EM, F1), system (avg latency, cost, avg chunks); existing helpers in `retrieval_metrics`, `answer_metrics`, `system_metrics`.  
- **run_ablations()**: Run one or more strategies (baseline, rules, learned, bandit_linucb, bandit_epsilon, bandit_reinforce) on the eval set, aggregate metrics, return table rows.  
- **format_table()**: Text table of strategy vs metrics.  
- **Script**: `phase4_run_eval.py --eval-set data/labels/eval_set.jsonl --checkpoint checkpoints/router`; optional `--strategies`, `--output`, `--format table|csv|json`.

**Implemented (optional)**  
- **Phase 0 index build**: `scripts/phase0_build_indexes.py` builds BM25 (rank_bm25), vector (sentence-transformers), DuckDB, and code indexes from `data/raw/*.jsonl` or `*.txt`; retrievers load from `data/processed/`.  
- **Pipeline LLM**: Pass `llm_client(query, context)` to `RAGPipeline`, or set `OPENAI_API_KEY` to use OpenAI; prompt built in `src/rag/llm.py`.

---

## Phases (overview)

- **Phase 0**: Setup data and indexes (pluggable retrievers, common schema, baseline RAG)
- **Phase 1**: Rules-based router baseline (keywords, centroids, logging)
- **Phase 2**: Supervised transformer router
- **Phase 3**: Bandit/RL optimization from online feedback
- **Phase 4**: Evaluation and ablations

## Structure

- `config/` — Settings, index definitions, router actions
- `data/` — Raw data, processed chunks, labels
- `src/` — Schema, retrievers, RAG pipeline, router (rules/learned/bandit), logging, evaluation, API
- `scripts/` — Phase entrypoints
- `tests/` — Unit tests
- `notebooks/` — Exploration and analysis

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY for pipeline answer generation
```

## Run

- Build indexes (Phase 0): `python scripts/phase0_build_indexes.py` (reads `data/raw/*.jsonl` or `*.txt`; writes to `data/processed/`. If raw missing, builds sample indexes.)
- Collect labels: `python scripts/phase1_collect_labels.py` (optional `--offline-eval` for routing labels)
- Train router (Phase 2): `python scripts/phase2_train_router.py` (requires `data/labels/routing_labels.jsonl`)
- Load learned router: `from src.router.learned import load_router; router = load_router("checkpoints/router")`
- Bandit (Phase 3): `python scripts/phase3_bandit_online.py --strategy linucb --mode replay` (or `reinforce` with `--checkpoint checkpoints/router`; use `--save` to write updated policy). Online: `--mode online --queries data/labels/queries.jsonl`.
- Run eval (Phase 4): `python scripts/phase4_run_eval.py` (uses `data/labels/eval_set.jsonl`; create with query, optional reference_answer, gold_doc_ids). Optional: `--output results.txt`, `--format csv|json`.
