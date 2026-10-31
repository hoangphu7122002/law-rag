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

## Step 8 — Dev 300 and the tuning method (2026-10-31)

- **Decided:** split test/dev/train = 200/300/2,245 (dev grew from 100 to cut selection noise). Tuning = grid search for
  discrete knobs + an autoresearch-style agent loop on `dev` only; a change is kept only if its gain passes a paired
  bootstrap test on `dev`. `test` is run once, by hand, at the end.

## Step 9 — First GPU runs on expiring Modal credit (2026-10-31)

- **Ran:** 4 jobs in parallel on 3 accounts (`scripts/vendors/modal/app.py`), ~4.0 USD, until each account's
  credit ran out. Log: `docs/ops/compute-log.md`. Outputs: `results/runs/20261031_modal/`.
- **Baseline on dev (300, vLLM, answer-only):** Qwen1.5-1.8B + course prompt, 256 tokens: cosine 0.592, Jaccard 0.163,
  token overlap 0.184, BLEU-1 0.126, ROUGE-L 0.155. With 768 tokens: token overlap 0.231, BLEU-1 0.116, ROUGE-L 0.132.
- **Prompt echo confirmed:** the course notebook pipeline returns prompt + answer on 25/25 test rows. Scored as-is (B0)
  token overlap is 0.778 vs 0.314 answer-only (B1); report both and use B1 as the fair baseline. The HF fp32 pipeline
  takes ~118 s/question, so the full B0/B1 rerun moves to vLLM or fp16 with the saved FAISS index.
- **CPT cost:** LoRA r64, seq 1024, one L4: 1.5B 1,537 tok/s, 3B 954 tok/s. 428 MB corpus ≈ 114M tokens →
  ~21 h (1.5B) / ~33 h (3B) per epoch. Full-corpus CPT on 3B is too costly; use 1.5B or a filtered 20–30M-token subset.
- **Embeddings cached** on the account-1 volume: c2000 × {bge-m3, Vietnamese_Embedding}, 188,228 chunks.
