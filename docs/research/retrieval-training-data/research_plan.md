# Plan: data + base models to beat AITeamVN (Vietnamese_Embedding / Vietnamese_Reranker) on Vietnamese legal retrieval
User (2026-11-19): after ReAct comes retrieval + reranker training; with current techniques and the right model we should surpass AITeamVN → research the training data (and retrieval if needed).
Builds on docs/notes/10-retrieval-training/research_report.md (R1 reranker FT, E1 embedder FT, S1 synthetic queries).
Subtopics
1. datasets_and_aiteamvn: public Vietnamese legal / general retrieval training sets (Zalo Legal 2021, BKAI, ALQAC, VLSP, mMARCO-vi, ViHERMES/VLQA, VN-MTEB...): size, licence, format, overlap risk; what AITeamVN trained on (model cards) and their reported scores.
2. base_models: stronger / complementary embedders and rerankers (Qwen3-Embedding, jina-v3/v4, mE5-instruct, gte-multilingual, BGE-reranker-v2-m3, Qwen3-Reranker, jina-reranker) with Vietnamese benchmark scores and sizes (2x L40S 46 GB).
3. recipes_and_eval: what gave the largest gains on legal retrieval (hard negatives, LLM synthetic queries + filtering, LLM-listwise distillation, LLM-based embedders with LoRA, bge-m3 multi-vector, domain-adaptive pretraining) and how to compare fairly with AITeamVN.
Synthesis: data inventory with decontamination risk, base-model shortlist, training plan and fair evaluation protocol.
