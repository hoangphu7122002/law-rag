# Plan: what practitioners do after SFT/RL for small-LLM legal RAG QA (no ReAct/agents)
Question: typical next steps when SFT variants plateau (law-rag: Qwen3.5-2B, test rougeL ~.62; errors = wrong Điều, retrieval misses).
Subtopics
1. generation_side: MBR / self-consistency / best-of-n selection for lexical metrics; rejection-sampling fine-tuning (RFT/expert iteration); model merging / ensembling; citation-aware training.
2. retrieval_side: embedding finetune, hybrid BM25+dense, query rewriting/expansion, reranker distillation, synthetic QA (doc2query) for retrieval; gains reported on legal corpora.
3. competitions: Zalo AI legal text retrieval 2021, ALQAC, VLSP legal QA, RAG challenges with ROUGE/BLEU metrics — winning recipes.
Synthesis: ranked list of next experiments with evidence strength, gaps.
