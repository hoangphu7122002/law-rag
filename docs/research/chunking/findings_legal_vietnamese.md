# Chunking for legal RAG: legal benchmarks + Vietnamese systems

Status: ALL items below are from search-result snippets only (3 WebSearch calls). No page was opened/read in full. Verify before citing numbers.
Legend: [SNIPPET] = seen in search summary only. [READ] = none.

## 1. LegalBench-RAG
- [SNIPPET] Benchmark retrieves minimal relevant snippets, not whole docs. Variants: LegalBench-RAG and -mini. Original paper compared Naive vs Recursive Text Character Split (RCTS) chunking, text-embedding-3-large, measuring precision/recall.
- [SNIPPET] PromptLayer blog: "more advanced chunking strategies, such as using a Recursive Text Character Splitter, significantly improve retrieval accuracy" (no numbers; secondary source).
- [SNIPPET] Follow-up arXiv 2508.13107 reports gains over that baseline: Recall@K +30-95%, Precision@K ~2.5x for K>4 (its own method, not chunking per se).
- Gap: exact RCTS vs naive P/R numbers NOT obtained. Need the original paper (arXiv 2408.10343, ID from memory, unverified).
- https://www.promptlayer.com/research-papers/legalbench-rag-a-benchmark-for-retrieval-augmented-generation-in-the-legal-domain
- https://arxiv.org/html/2508.13107v2

## 2. Structure-aware vs fixed-size (statutes)
- [SNIPPET] "Chunking German Legal Code" (BGB, 2026): chunking aligned with legal structure (section/subsection) gets highest recall; approaches overriding structure do worse. Fixed-size baseline splits provisions across overlapping windows. Scored at parent-section level. https://arxiv.org/pdf/2605.19806
- [SNIPPET] 36-strategy, 6-domain benchmark (incl. Legal): structurally informed/adaptive chunking often beats fixed-size, more robust; fixed-size char baselines poor; smaller chunks raise index size/latency. https://arxiv.org/pdf/2603.06976 (unsure which of the two IDs 2603.06976 / 2603.25333 is this)
- [SNIPPET] Adaptive chunking paper (legal/technical/social): answer correctness 72% vs 62-64% baseline; for legal, respect article delimiters even when articles vary in length. https://arxiv.org/abs/2603.25333
- [SNIPPET] Recursive splitting on legal docs: ~2-8% boundary alignment with section delimiters, 0/42 cross-reference preservation (one corpus). Source: arXiv 2610.26313 StrucChunk (https://arxiv.org/pdf/2610.26313), attribution uncertain.
- [SNIPPET] Thai legal NLLP 2026: 19 configs over chunk size, boundary strategy, contextual chunks. https://aclanthology.org/2026.nllp-1.19/
- Gap: no clean article-level vs fixed-window ablation on non-German corpus found.

## 3. Vietnamese legal RAG
- [SNIPPET] VLSP 2025 Hoai et al., hierarchical chunking: leaf nodes >4000 tokens re-split into 2048-token chunks, 1024 overlap. ~2000+ legal docs segmented into articles, often lacking title metadata (hurts retrieval). https://aclanthology.org/2025.vlsp-1.18
- [SNIPPET] VIDRILL (VLSP 2025): dual-level chunking, long chunks up to 2000 chars for rerank, fine chunks capped at 450; plus two-stage hard-negative mining. Likely https://preview.aclanthology.org/vol-to-evt/2025.vlsp-1.17.pdf (or 1.20; unconfirmed).
- [SNIPPET] HF repo AnTrc2/R2AI2026-BUILD-AI-LEGAL-ASSISTANT: each leaf (Dieu/Khoan/Diem) = one chunk, text prefixed with doc name + article no. + heading hierarchy; Qwen3-Embedding-0.6B + FAISS. https://huggingface.co/AnTrc2/R2AI2026-BUILD-AI-LEGAL-ASSISTANT
- [SNIPPET] Other Vietnamese: arXiv 2409.13699 (Vietnamese Legal IR in QA), VLQA dataset 2507.19995, SCITEPRESS 137512. Chunk details unseen.
- Gap: no source with retrieval metrics for per-Dieu vs fixed-size on Vietnamese data.

## Takeaways for law-rag
- Chunk by structure (Dieu, split to Khoan only if long), prefix doc name + Dieu number + heading path (contextual header) is common practice.
- Cap long Dieu with overlapping sub-chunks (VLSP: 2048/1024 tokens; VIDRILL: 450 chars fine level).
- Our own ablation (per-Dieu vs fixed windows) would fill an open literature gap.
