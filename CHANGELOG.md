# CHANGELOG

## Step 1 — Read the assignment, scaffold the repo (2026-10-31)

- **Did:** read RAG_Assignment.pdf, Baseline.ipynb, Dataset.zip (16,441 law files), RES sheet (2,745 QA pairs);
  wrote the brief, a line-for-line port of the baseline metrics with tests, and the repo layout.
- **Next:** research directions (hybrid retrieval, structure-aware chunking, Graph RAG, LoRA on a < 4B model),
  then the baseline re-run on the fixed test split.

## Step 2 — `make data` and the first data finding (2026-10-31)

- **Did:** `make data` → 16,440 corpus files, 2,745 QA pairs, split test/dev/train 200/100/2,445 (seed 20261031).
- **Finding:** the QA set mostly asks about laws that are **not in the corpus**. 772 rows have no law id
  ("Không trích xuất được"); of the 1,973 with an id (314 distinct, mostly 2024–2025 banking circulars such as
  52/2024/NĐ-CP, 17/2024/TT-NHNN), only 8 ids / 51 rows match a corpus file name, and the top 5 ids appear in no
  corpus file text. Retrieval over the corpus alone cannot find the gold article for most questions.

## Step 3 — Research directions and the train-overlap finding (2026-10-31)

- **Finding:** 93 % of test questions with a law id cite a law also present in `train`, 57 % the same (law, Điều).
- **Research:** ranked directions in `docs/notes/01-research-directions.md`: LoRA/RAFT SFT on `train`, train-QA
  memory, hybrid + Điều chunking + reranker, a cheap regex law graph as the Graph RAG part. Default generator
  Qwen2.5-3B-Instruct (3.09B); Qwen3-4B (4.0B) is not under 4B.

## Step 4 — Roadmap settled (2026-10-31)

- **Decided:** due 2026-11-29; compute = Modal 1× L4 only. Roadmap in `docs/plan/roadmap.md`: core items, if-time items
  (DPO, GRPO, extra layers, LightRAG, …) and skipped items with reasons (verl multi-GPU, MS GraphRAG no delete,
  Graphiti/Mem0 need strong JSON LLMs, xRAG too little data, web lookup leaks answers).

## Step 5 — Roadmap widened to 90 USD, multi-L4 allowed (2026-10-31)

- **Decided:** 3 Modal accounts (~90 USD), multi-L4 allowed; almost every idea is planned. Risky ones run with guards
  (offline large extractor, anti-copy reward, verl timebox). Skipped only: web lookup at answer time, crawling Q&A
  sites (answer leakage), xRAG, Self-RAG, answering with a model ≥ 4B.

## Step 6 — Continued pretraining on a large crawled legal corpus (2026-10-31)

- **Decided:** crawl a large legal pretraining corpus (D10, all official texts, target ≥ 300M tokens) and legal Q&A pages
  (D11, guarded by 13-gram decontamination vs test/dev, own data version). Continued pretraining P-1: LoRA-CPT,
  extra layers, full CPT of a ≤ 1.7B model on 2–4 L4, then SFT on top.

## Step 7 — Rule: no test knowledge in any trained weights (2026-10-31)

- **Decided:** training data (pretraining, SFT/RAFT/DPO/GRPO, embedding fine-tune) excludes test/dev QA, train pairs
  citing the same (law, Điều) as test/dev, anything sharing a non-boilerplate 13-gram with test/dev answers, and the
  crawled text of test/dev-cited articles. The RAG store may hold everything except the test/dev QA pairs.
- **Measured on train (test only):** same (law, Điều) removes 16 % (keeps 2,050 / 2,445); rare 13-gram (in ≤ 3 answers)
  removes 20 % (keeps 1,966); dropping whole laws would keep only 57 % (too strict, not used). Dev filter still to measure.
- **Amended:** the knowledge filter is against `test` only. `dev` pairs are held out while tuning, then the final
  model is retrained on train + dev (also filtered against test).
