# Roadmap — Vietnamese Law RAG

**Due: 2026-11-30 00:00 → submit by 2026-11-29.** Deliverables: source code, dataset, manual (reproducible), report, demo.
**Compute:** Modal, **1× L4 (24 GB) only**, same runner style as Triton; ~30 USD (one account after the 11-01 reset).
Laptop (Mac MPS) for everything that fits. Every run logged in `docs/ops/compute-log.md`.

**Method rule:** change one axis at a time, keep the best, move on. Tune on `dev` (100), report on `test` (200).
Metrics: the 5 baseline metrics + law hit@k + latency + cost per question, broken down by question group
(law in corpus or not, has law id or not, time-dependent or not).

Research behind each choice: `docs/notes/01`–`04`.

## Schedule

| Week | Do | Output for the report |
|---|---|---|
| **W1** 10-31 → 11-08 (Big Data slides 11-08, keep light) | Baseline B0–B3 · EDA · crawl pilot (20 missing laws) · pipeline as modules + YAML config from day one | Baseline numbers, failure list, data facts |
| **W2** 11-09 → 11-15 | Dataset D1, D3, D7–D9 · chunking · stores · retrieval | Dataset v1/v2 table, store × retrieval table (hit@k) |
| **W3** 11-16 → 11-22 | Query rewrite · models · prompt · copy-bias · SFT → RAFT | Rewrite table, model table, training table |
| **W4** 11-23 → 11-29 | Agents + conversation · final combo on `test` · docker compose · demo · report | Final table baseline → best, demo, manual |

## Do (core)

| Group | Items |
|---|---|
| 0 Baseline | **B0** as-is · **B1** answer-only scoring (check if output contains the prompt) · **B2** gold-context ceiling · **B3** copy nearest train answer |
| 1 Data | **D1** crawl missing laws (.gov sources only) · **D3** versions + amends/replaces/repeals + effective dates · **D7** clean, keep structure, metadata · **D8** decontaminate against `test` · **D9** dataset versions v0/v1/v2 on HF |
| 2 Chunking | **C1** 2000 chars · **C2** 300–500 tokens · **C3** by Điều/Khoản · **C4** parent–child · **C5** heading prefix |
| 3 Stores | **K1** flat vectors (Qdrant) · **K3** Neo4j graph built by rules (MERGE: easy insert/update/delete) · **K4** temporal graph ("law as of date X", change history) · **K8** train QA memory · **K10** metadata filter |
| 4 Retrieval | **R1** embeddings (sbert, bge-m3, Vietnamese_Embedding, e5) · **R2** BM25 · **R3** hybrid RRF · **R4** reranker · **R7** graph expansion |
| 5 Query rewrite | **Q0** none · **Q1** normalize abbreviations · **Q2** multi-query + RRF · **Q5** decomposition · **Q6** extract law id / article / date |
| 6 Models + prompt | **M1** Qwen1.5-1.8B · **M2** Qwen2.5-1.5B/3B · **M3** Qwen3-1.7B · **M5** Vi-Qwen2-3B-RAG · **P1** "Căn cứ…" template · **P2** 1–2 nearest train examples · **P3** length 256/512/768 |
| 7 Model changes | **A1** copy-bias logits (no training) · **T1** SFT LoRA · **T2** RAFT · **T7** compare base → SFT → RAFT |
| 8 Agents + conversation | **H1** fixed LangGraph flow: router → rewriter → retrieve → generate → verifier · **H2** history-aware rewrite · **H3** slots (law/article/date in focus) · **H6** 20–30 multi-turn test chains · **H7** guided JSON + rule fallback · **H8** check law still in force · **G1** router by law id · **G2** CRAG fallback · **G4** verifier (format, cited article exists) |
| 9 Production | **X1** one code path, YAML config · **X2** docker compose (FastAPI, vLLM, Qdrant, Neo4j, Gradio) · **X4** citations with links |

## If time and credit remain (in this order)

| Item | Why it is worth it | Condition |
|---|---|---|
| **T3** DPO (TRL, LoRA) | cheap step after SFT, low risk | after T2, ~1–3 USD |
| **T4** GRPO (Unsloth/TRL, 1 L4) reward = metrics + anti-copy + length | trains directly on the grading metrics | Qwen3-1.7B first; 10-step timing trial; ~4–9 USD |
| **A4** extra layers (LLaMA-Pro) on crawled laws | the "change the architecture" idea | after T2 works; short sequences to fit 24 GB |
| **K5** LightRAG on a subset | compare with our rule graph | extractor = best < 4B model; expect weak extraction |
| **R10** fine-tune embedding on train pairs | often a large retrieval gain | cheap on L4 |
| **D2** 2025–2026 laws · **D4** consolidated texts (VBHN) | bigger, newer dataset | crawler works |
| **Q3** HyDE · **Q4** step-back · **P4** verbatim quoting · **R8** context order | cheap prompt-level ablations | any time |
| **H4** session summary + user facts · **G5** refuse when no basis · **X3** incremental crawler · **X5** /metrics · **X6** CI eval gate | production polish | W4 |
| **M4** Llama-3.2-3B, Gemma · **D5** Zalo/ALQAC/VLQA · **K9** doc2query | more comparison rows | spare time |

## Skip (and why)

| Item | Reason |
|---|---|
| verl (T6) | built for multi-GPU (the 3B example uses 2–4 A100); we train on 1 L4; TRL/Unsloth do GRPO on one GPU |
| Microsoft GraphRAG (K6) | heaviest LLM indexing and **no delete**; fails the insert/update/delete requirement |
| Graphiti (K7), Mem0 (H5) as main store | need reliable JSON from a strong LLM (< 4B models emit strict JSON ~5/1000 raw); vendor benchmarks conflict |
| xRAG projector (A5) | needs far more than 2.4k training pairs; high risk, no time |
| Continued pretraining (A8) | forgets general skills; A4 does the same job safer |
| Self-RAG (G3) | needs special reflection tokens trained in; G2 CRAG gives the same idea cheaply |
| Web lookup at answer time (G6) | leaks reference answers (Q&A sites), slow, not reproducible; crawl offline instead (D1) |
| Trained query rewriter (Q8), query2doc (Q7), MMR (R6), compression (R9), semantic chunking (C6), tree store (K2), constrained citations (A6), DPO variants ORPO/KTO, Llama-Pro + RL combos | small or duplicate gain for the time; covered by a simpler item above |
| General-knowledge corpus (D6) | metrics are about specific law articles; off-target |
| Crawling Q&A websites | likely contains the reference answers → inflated, unfair scores |

## Budget (1× L4, Modal)

| Part | USD |
|---|---|
| Baseline, retrieval, rewrite, model sweep | ~4–6 |
| SFT + RAFT | ~2–3 |
| If-time: DPO, GRPO, extra layers, LightRAG | ~8–18 |
| Agents, final combo, demo | ~2–3 |
| **Core total** | **~8–12** · with if-time items ~20–30 |

## Cut order if late

If-time items first, then H6 multi-turn set → K4 temporal graph polish → docker (keep a plain `make demo`).
Never cut: baseline, dataset, one store comparison, SFT/RAFT, final table, manual.
