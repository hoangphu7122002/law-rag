# Project brief — RAG Assignment (Building a Simple Law Chatbot)

Source: `RAG_Assignment.pdf` in this folder (Tho Quan, v1.0, 2025-11-09) and the course Drive folder
"RAG Assignment" (Baseline.ipynb, Dataset.zip, RES sheet).

## Task

Improve a naive RAG baseline for Vietnamese legal QA. Focus: retrieval and how retrieved context is used in
generation. Free choice of embedding model; free choice of LLM **under 4B parameters**.

## Baseline (what we must beat)

| Part | Baseline |
|---|---|
| Chunking | `RecursiveCharacterTextSplitter`, 2000 chars, overlap 50; text lower-cased, newlines removed |
| Embedding | `keepitreal/vietnamese-sbert` |
| Vector DB | FAISS (LangChain), top k = 5 |
| LLM | `Qwen/Qwen1.5-1.8B` (base, not chat), `max_new_tokens=256` |
| Prompt | "Bạn là một trợ lý thông minh ... Trả lời ngắn gọn và chính xác nhất có thể" |
| Eval | `df.head(30)` in the notebook — the assignment requires **≥ 100** pairs |

## Data (measured 2026-10-31)

- **Corpus:** `Dataset.zip` → `export_1/`, **16,441** `.txt` files, 428 MB. File name = law number with `/`
  replaced by `_` (e.g. `08_2024_TT-BLĐTBXH.txt`; the zip stores `Đ` badly, shown as `??`).
- **RES (QA):** 2,745 rows. Columns: `Câu Hỏi` (question, median 137 chars), `Trả lời` (reference answer,
  median 1,426 chars, max 9,858), `Số hiệu VBPL (Trích xuất)` (law number cited, e.g. `50/2024/TT-NHNN`),
  `Câu trả lời của URAx-Law` (another system's answer), two 0/1 check columns, and junk summary columns.
- Reference answers follow a template: *"Căn cứ tại khoản 2 Điều 17 Thông tư 50/2024/TT-NHNN quy định ..."*
  then a near-verbatim quote of the article. This matters: lexical metrics reward quoting the right article.

## Metrics (all on normalized text: lower, strip punctuation)

Cosine (embedding of answer vs reference, baseline uses vietnamese-sbert), Jaccard, Token overlap
(recall of reference tokens), unigram BLEU with brevity penalty, ROUGE-L (LCS F1). Exact code:
`src/evaluate/metrics.py`.

## Grading (10 pts)

| Item | Pts |
|---|---|
| Improvement over baseline on ≥ 1 metric | 3 |
| Demo on a few example questions (clear, correct, runs without errors) | 2 |
| Report: data exploration 1 · method vs baseline with numbers 3 · format, clarity, reproducible instructions 1 | 5 |

Submission: source code, dataset, manual (reproduce steps).
