# Research plan: model choice for law-rag (2026-11-17)

**Question:** Which instruct LLM (< 4B params, strict) and which embedding + reranker are best for Vietnamese legal RAG,
scored by lexical metrics (BLEU-1, ROUGE-L, token overlap, Jaccard) + sbert cosine; answers ~1,400 chars,
"Căn cứ tại khoản X Điều Y ..." template; served by vLLM on L40S 46 GB.

Already known (docs/notes/01): AITeamVN Vietnamese_Embedding Zalo legal Acc@1 0.727 vs bge-m3 0.568 (self-reported);
Qwen2.5-3B-Instruct 3.09B; Qwen3-1.7B; Vi-Qwen2-3B-RAG; ViLegalQwen3-1.7B (no scores); Qwen3-4B excluded.

## Subtopics
1. **LLM** — instruct models < 4B as of Nov 2026 (Qwen2.5/3/3.5+, Gemma 3/newer, Llama 3.2, Phi, SmolLM, Vietnamese fine-tunes):
   exact param count, Vietnamese quality evidence (VMLU, ViLLM-Eval, legal), context length, license, vLLM support.
2. **Embedding** — Vietnamese retrieval embedders (Vietnamese_Embedding, bge-m3, Qwen3-Embedding 0.6B/4B/8B,
   multilingual-e5, halong, gte, newer): VMTEB / Zalo legal / ALQAC numbers, max length, size.
3. **Reranker** — Vietnamese rerankers (AITeamVN Vietnamese_Reranker, bge-reranker-v2-m3, Qwen3-Reranker, others):
   legal retrieval numbers, latency/size.

## Synthesis
One shortlist per component (default + 1–2 to ablate), with evidence strength, into research_report.md.
