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
