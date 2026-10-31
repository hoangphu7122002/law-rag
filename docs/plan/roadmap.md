# Roadmap — Vietnamese Law RAG

**Due: 2026-11-30 00:00 → submit by 2026-11-29.** Deliverables: source code, dataset, manual (reproducible), report, demo.
**Compute:** Modal, 3 accounts ≈ **90 USD**. Default 1× L4 (24 GB); multi-L4 (2–4) allowed when a job needs it.
Mac (MPS) for what fits. Every run logged in `docs/ops/compute-log.md`.
**Real limit = time (4 weeks, solo, Big Data in parallel), not money.** Parallel agent PRs like Triton.

**Method rule:** change one axis at a time, keep the best, move on. Tune on `dev` (300), report on `test` (200).
Metrics: 5 baseline metrics + law hit@k + latency + cost/question, split by question group
(law in corpus or not, has law id or not, time-dependent or not).

**Model rule:** the answering LLM is always **< 4B**. A larger model (e.g. Qwen3-30B-A3B on 2× L4) may be used
**offline only** (graph extraction, data generation, judge) and is reported as such. Confirm with the lecturer.

Research behind each choice: `docs/notes/01`–`04`.

## Schedule

| Week | Do |
|---|---|
| **W1** 10-31 → 11-08 (Big Data slides 11-08) | Group 0 baseline · EDA · start the crawler (runs in background all month) · pipeline as modules + YAML |
| **W2** 11-09 → 11-15 | Groups 1–4: dataset, chunking, stores (incl. graph variants), retrieval · continued pretraining starts on the crawled corpus |
| **W3** 11-16 → 11-22 | Groups 5–7: query rewrite, models, prompt, copy-bias, SFT → RAFT → DPO → GRPO, extra layers |
| **W4** 11-23 → 11-29 | Group 8–9: agents + conversation, final combo on `test`, docker, demo, report |

## Tuning method

- **Grid search** for discrete knobs (k, chunk size, embedding, reranker, LR, LoRA rank, epochs).
- **Agent tuning loop** (Karpathy autoresearch style) for open spaces (prompts, query rewrite, context packing,
  GRPO reward, verifier rules): fixed `eval_dev.py` (agent cannot edit, loads `dev` only) → agent edits only the
  pipeline config / prompts → run with a time + cost cap → keep if better, revert if not → log every run to `results.tsv`.
- **Guards:** the agent never sees `test` (human runs it once at the end); one objective = mean of the 5 metrics on
  `dev`, all 5 logged; a change is kept only if the gain beats noise (paired bootstrap on `dev`, 95 %);
  per-run and per-night caps; repo hard rules still apply.

## Experiments (all planned; ◇ = run last, cut first if late)

| Group | Items |
|---|---|
| 0 Baseline | B0 as-is · B1 answer-only scoring · B2 gold-context ceiling · B3 copy nearest train answer |
| 1 Data | D1 missing laws (.gov) · D2 2025–2026 laws · D10 large legal pretraining corpus (all official texts, gazette, all years; target ≥ 300M tokens) · D11 legal Q&A pages (guarded, below) · D3 versions + amends/replaces/repeals + effective dates · D4 consolidated texts (VBHN) · D5 Zalo/ALQAC/VLQA/BKAI · D7 clean + metadata · D8 decontaminate vs `test` · D9 versions v0–v3 on HF · ◇ D6 general knowledge |
| 2 Chunking | C1 2000 chars · C2 300–500 tokens · C3 Điều/Khoản · C4 parent–child · C5 heading prefix · C6 semantic |
| 3 Stores | K1 flat vectors (Qdrant) · K2 tree · K3 Neo4j rule graph (MERGE insert/update/delete) · K4 temporal graph · K5 LightRAG · K6 MS GraphRAG (subset, comparison only) · K7 Graphiti (temporal) · K8 train QA memory · K9 doc2query · K10 metadata filter |
| 4 Retrieval | R1 embeddings (sbert, bge-m3, Vietnamese_Embedding, e5) · R2 BM25 · R3 hybrid RRF · R4 reranker · R5 k · R6 MMR · R7 graph expansion · R8 context order · R9 compression · R10 fine-tuned embedding |
| 5 Query rewrite | Q0 none · Q1 normalize · Q2 multi-query + RRF · Q3 HyDE · Q4 step-back · Q5 decomposition · Q6 law id/article/date extraction · Q7 query2doc · ◇ Q8 trained rewriter |
| 6 Models + prompt | M1 Qwen1.5-1.8B · M2 Qwen2.5-1.5B/3B · M3 Qwen3-1.7B · M4 Llama-3.2-3B, Gemma < 4B · M5 Vi-Qwen2-3B-RAG, ViLegalQwen · P1 "Căn cứ…" template · P2 nearest examples · P3 length · P4 verbatim quoting |
| 7 Model changes + pretrain + post-train | **P-1 continued pretraining** on D10 (+D11): (a) LoRA-CPT, (b) A4 extra layers (LLaMA-Pro), (c) full CPT of a ≤ 1.7B model on 2–4 L4 (FSDP) — then SFT on top · A1 copy-bias logits · A6 constrained citations · T1 SFT LoRA · T2 RAFT · T3 DPO · T4 GRPO (reward = metrics + citation − copy − length) · T6 verl GRPO on 2–4 L4 (vs TRL/Unsloth) · T7 compare all |
| 8 Agents + conversation | H1 fixed LangGraph flow · H2 history-aware rewrite · H3 slots · H4 session summary + user facts · H5 Mem0/Graphiti memory (vs H4) · H6 20–30 multi-turn chains · H7 guided JSON + fallback · H8 in-force check · G1 router · G2 CRAG · G4 verifier · G5 refuse when no basis · G7 time agent |
| 9 Production | X1 one code path · X2 docker compose (FastAPI, vLLM, Qdrant, Neo4j, Gradio) · X3 incremental crawler · X4 citations · X5 /metrics · X6 CI eval gate |

## Do with a guard (risk known, timebox)

| Item | Risk | Guard |
|---|---|---|
| K5 LightRAG, K6 MS GraphRAG, K7 Graphiti, H5 Mem0 | < 4B extractors break JSON; MS GraphRAG has no delete | extract with the offline large model on a subset; MS GraphRAG as comparison only, not production |
| T4 GRPO | reward hacking (copy whole articles, padding), collapse | anti-copy + length terms; 10-step timing trial; early stop |
| T6 verl | heavy setup (Ray, FSDP), multi-GPU | timebox 1 day; if not running → TRL/Unsloth result stands |
| P-1 continued pretraining (LoRA-CPT, extra layers, full CPT) | forgetting, 24 GB memory | short sequences; compare with T2 on the same test |
| R10 embedding fine-tune, Q8 trained rewriter | overfit to 2.4k train | evaluate on dev only, report on test once |
| D11 crawl legal Q&A / advice pages | may contain the `test` reference answers → inflated scores; some sites forbid scraping | sites that allow it; 13-gram decontamination vs `test` (+ `dev`); kept as its own data version; report with / without |

## Skip (risky, and why)

| Item | Why |
|---|---|
| G6 web lookup at answer time | leaks reference answers from Q&A sites, slow, not reproducible → crawl offline instead (D1) |
| A5 xRAG projector | needs far more aligned data than 2.4k pairs; high chance of no result in 4 weeks |
| G3 Self-RAG | needs reflection tokens trained into the model; G2 CRAG covers the idea |
| Answering with a model ≥ 4B (incl. Qwen3-4B = 4.0B) | breaks the assignment rule |

## Budget (≈ 90 USD, 3 accounts)

| Account | Use | USD |
|---|---|---|
| 1 | baseline, retrieval, embeddings, query rewrite, model sweep | ~15–20 |
| 2 | continued pretraining (LoRA-CPT, extra layers, full CPT), SFT, RAFT, DPO, GRPO | ~35–45 |
| 3 | offline large model (graph extraction: LightRAG/MS GraphRAG/Graphiti/Mem0), verl, final runs, reserve | ~20–30 |

## Cut order if late

◇ items → T6 verl → K6 MS GraphRAG → P-1(c) full CPT → H5 → K7 → docker polish (keep `make demo`).
Never cut: baseline, dataset, store comparison, SFT/RAFT, final table, manual.
