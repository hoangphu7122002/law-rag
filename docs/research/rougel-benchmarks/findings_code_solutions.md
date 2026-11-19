# Findings: code/blog solutions for Vietnamese legal RAG (ROUGE-L)

Date: 2026-11-20. 5 searches + 1 fetch. No repo for the exact assignment found.

## Exact assignment (Qwen1.5-1.8B + vietnamese-sbert + FAISS + NHNN Thong tu)
- No GitHub/Kaggle/HF repo, README, or blog found. Queries tried: assignment stack, "Dataset.zip export_1 RES"-style, Thong tu NHNN. Zero hits. (Search engine is US-only/general; direct GitHub search not available.)

## Candidates
| Source | Stack | Metric | Value | Status |
|---|---|---|---|---|
| Hien, Nhi, Huyen 2025, SCITEPRESS https://www.scitepress.org/publishedPapers/2025/137512/pdf/index.html | GPT-4, Milvus, Vietnamese BERT 768d (unnamed), 45k legal docs, 2000-pair eval set vs expert answers | ROUGE-L (variant P/R/F unspecified) | 0.756 (ROUGE-1 0.834); baselines ROUGE-1: std RAG 0.687, Legal BERT-QA 0.723; baseline ROUGE-L not reported | VERIFIED in page (fetched). Caveat: GPT-4, expert-written refs, paper not reproducible, no code |
| VietLegal NeurIPS-2026 slides https://neurips.cc/media/neurips-2026/Slides/139241.pdf | agentic RAG, 55k civil-law statutes | ROUGE-L | no number seen; says ROUGE-L alone misleading | UNVERIFIED (snippet) |
| arXiv 2409.13699 Vietnamese Legal IR in QA (has GitHub link) | hybrid keyword+vector, RRF | retrieval metrics | n/a ROUGE-L | UNVERIFIED (snippet), not generation |
| ntphuc149/ViLegalLM https://github.com/ntphuc149/ViLegalLM | ViLegalQwen2.5-1.5B / Qwen3-1.7B base, ViLegalBERT | n/a | n/a | UNVERIFIED (snippet); no RAG pipeline seen |
| HF space YuITC/Vietnamese-Legal-Doc-Retrieval | FAISS + multilingual BERT legal ft | none | - | UNVERIFIED (snippet) |
| HF AnTrc2/R2AI2026-BUILD-AI-LEGAL-ASSISTANT | FAISS + Qwen3-Embedding-0.6B | none | - | UNVERIFIED (snippet) |
| arXiv 2403.01616 Vietnamese RAG benchmark | Vietnamese-SBERT retriever, general domain | n/a | - | UNVERIFIED (snippet) |
| Mao et al. Dec 2025 hybrid legal RAG agent (Law_QA, not Vietnamese) | - | F1, ROUGE-L | gains over vanilla, no number | UNVERIFIED (snippet) |

## Conclusions
- Highest ROUGE-L found: 0.756 (SCITEPRESS 2025), GPT-4 + Milvus, expert refs. Not comparable to 1.8B base-model assignment.
- No report of 0.8-0.9 ROUGE-L found anywhere.
- No public solution for this assignment found; any 0.7-0.9 would likely come from near-copy of retrieved legal text / template-matching refs ("Can cu tai khoan X Dieu Y ... quy dinh ..."), not verified by any source.
