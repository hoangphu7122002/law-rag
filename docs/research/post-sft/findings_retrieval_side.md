# Findings: retrieval side (agent summary, snippets only; unverified numbers)
- Vietnamese legal hard negatives + reranker (arXiv 2507.14619): fine-tuned bi-encoder beat BM25, Exist@90 97 %; semi-hard negatives mined from the bi-encoder → +23 % relative reranker; reranker MRR@10 79.11 % eval / 77.54 % private test.
- BKAI synthetic queries (arXiv 2412.00657): Llama 3, ~500k synthetic queries from legal passages for bi-encoder + ColBERT pretraining then contrastive FT with mined hard negatives; "significant" in-domain gains TVPL / Zalo Legal 21 (numbers not seen).
- VLSP 2025 DRiLL (aclanthology 2025.vlsp-1.20): BM25 + bge-m3 hybrid, bge-reranker-v2-m3 fine-tuned with hard negatives. VIDRILL (2025.vlsp-1.17): BM25 + dense + reranker, dual-level chunking, two-stage hard-negative mining.
- ICTA 2023: hard negatives from BM25/TF-IDF; refresh with the current model during training.
- Brazilian case law (ENIAC): HyDE vs dense Recall@1 .086 → .345; HyDE + BM25 RRF MRR .571 ≈ pure BM25 .574 → BM25 very strong on legal text, fusion adds little over it.
- Chinese statutes LegalMALR: multi LLM query reformulations + dense per reformulation + merge + zero-shot LLM reranker (118 queries only). Public-defender retrieval (arXiv 2601.14348): fine-tuning on existing legal benchmarks HURT; filtered synthetic data + domain query generator + legal reranker helped.
- Practitioner: hybrid + RRF, cross-encoder, hard metadata filters when the query names a section (anecdotal).
- Takeaways: hard negatives most consistent; filtered synthetic queries help, generic legal FT can hurt (decontaminate, hold out dev); hybrid gains modest; HyDE/expansion large vs dense-only but shrink with BM25; reranker distillation: no legal evidence found (gap).
