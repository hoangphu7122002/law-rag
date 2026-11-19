# Plan: which models fit Vietnamese legal RAG (user, 2026-11-19: "nghiên cứu model phù hợp nữa")
Previous round (docs/research/retrieval-training-data) found no Vietnamese-legal scores for most candidates; this round hunts for ACTUAL numbers.
Subtopics
1. embedders: VN-MTEB leaderboard / MTEB-vi retrieval / Zalo legal / MIRACL-vi numbers for Qwen3-Embedding, bge-m3, AITeamVN, halong, Vietnamese bi-encoders, mE5, gte-multilingual, jina, Arctic, NV-Embed; long-article handling, instruction prefixes, sparse / multi-vector.
2. rerankers: Vietnamese numbers for AITeamVN Vietnamese_Reranker, PhoRanker, bge-reranker-v2-m3, Qwen3-Reranker, jina, mxbai, gte-multilingual-reranker, LLM listwise rerankers.
3. answering_llms: <4B models for Vietnamese legal RAG (Qwen3.5-2B / Qwen3-1.7B / Qwen2.5-3B, Gemma 3, Llama-3.2-3B, Phi-4-mini, SmolLM3, Vi-Qwen2-3B-RAG, ViLegalQwen, Vistral, Sailor2, VyLinh, SeaLLM): exact parameter counts, Vietnamese benchmarks (VMLU, ViMMLU, SeaExam), licences.
Synthesis: shortlist per role with evidence strength, what to test first on our dev_clean / dev_strict.
