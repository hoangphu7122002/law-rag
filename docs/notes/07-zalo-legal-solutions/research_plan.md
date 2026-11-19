# Research plan: Zalo AI legal solutions and Vietnamese legal RAG (2026-11-18)

**Question:** What do top Vietnamese legal retrieval / QA solutions (Zalo AI Challenge 2021 Legal Text Retrieval,
later Vietnamese legal RAG repos and HuggingFace models) do in their architecture, and what can law-rag borrow
(Vietnamese law QA, LLM < 4B, metrics cosine / Jaccard / token overlap / BLEU-1 / ROUGE-L)?

## Subtopics

1. **Zalo AI 2021 Legal Text Retrieval winners** — top-3 team write-ups and GitHub repos: data, BM25 + dense
   hybrid, Condenser / co-Condenser pre-training, hard negatives, ensembling, scores.
2. **Vietnamese legal retrieval models on HuggingFace** — embedders and rerankers trained on Zalo legal / BKAI
   legal data (AITeamVN, bkai-foundation-models, dangvantuan, truro7, hiieu …): training recipe, benchmarks.
3. **Vietnamese legal QA / RAG systems with generation** — open repos and papers (Vi-Qwen2-RAG, ViLawQA,
   ALQAC, VLSP legal tasks, chatbot projects): chunking by Điều, metadata, graph, small-LLM fine-tuning, eval.

## Output

`findings_<subtopic>.md` per subtopic (facts, numbers, URLs), then `research_report.md`: what is special,
what to borrow, mapped to the law-rag roadmap phases.
