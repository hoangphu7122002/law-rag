# Model selection for Vietnamese legal RAG — what the evidence supports (2026-11-19)

Sources: findings_embedders.md, findings_rerankers.md, findings_answering_llms.md. Evidence is thin; nothing below is a measurement on our data.

## Embedder (candidate generation; recall is not our bottleneck: Điều@5 .90, @20 .97)
- Keep AITeamVN/Vietnamese_Embedding for now: it is the legal fine-tune that brought bge-m3 from MRR@10 .682 to .818 on Zalo, although it is BELOW bge-m3 on general VN-MTEB retrieval (34.18 vs 39.84) → its strength is legal-domain only (and Zalo is in its training set).
- Challengers by evidence: multilingual-e5-large-instruct (VN-MTEB 40.88, best sub-1B; 512-token limit hurts long articles), gte-Qwen2-1.5B-instruct (42.01, 1.5B), Qwen3-Embedding 0.6B/4B (32K, no Vietnamese numbers, hypothesis). None has a Vietnamese-legal number.
- Cheap test that needs no re-index: re-score the existing top-50 candidates of dev_clean with each challenger (≈ 8k embeddings) and compare ordering with the current embedder on the same pool.

## Reranker (where the gain is: rr_v0 moved Điều@1 .587 → .712)
- Strongest published Vietnamese numbers: AITeamVN Vietnamese_Reranker (Zalo MRR@10 .867, in-domain); PhoRanker (MMARCO-vi NDCG@10 .742, 0.1B, 256 tokens); ViRanker (.730, self-reported); bge-reranker-v2-m3 .687 on the same set.
- Challenger with the best multilingual evidence: Qwen3-Reranker-4B (MMTEB-R 72.74; 8B adds little), LoRA, 32K ctx, slower than encoders; no Vietnamese-specific numbers. jina rerankers are non-commercial (skip).
- Plan: fine-tune the AITeamVN reranker again as the main line (rr_v0 exists); add Qwen3-Reranker-4B LoRA and raw bge-reranker-v2-m3 as rows in the same table.

## Answering LLM (< 4B)
- Current choice Qwen3.5-2B (2.27B measured from safetensors) is consistent with the evidence available; no source ranks the < 4B candidates on Vietnamese legal QA or RAG faithfulness.
- Qwen3-4B (4.0B total, 3.6B non-embedding) sits at the limit: only usable if the exact safetensors total is < 4.0B; roadmap already excludes it. Gemma 3 4B counts the vision tower; Sailor2-3B metadata says 4B; Vi-Qwen2-3B-RAG card is inconsistent.
- Measure ourselves (SFT on the same data, same retrieval, dev_clean / dev_strict): Qwen3.5-2B (have), Vi-Qwen2-3B (done: .5986 dev vs .6026), Qwen3-1.7B, Qwen2.5-3B (licence non-commercial: check), Llama-3.2-3B, Phi-4-mini 3.8B, SmolLM3-3B. Earlier results show the base model matters little next to SFT style and MBR.

## What to test first (cheap → expensive)
1. rr_v0 with k = 1–3 in generation (no training).
2. Candidate re-scoring of the top-50 with Qwen3-Embedding-0.6B / mE5-instruct / gte-Qwen2-1.5B (no index rebuild).
3. Reranker table: AITeamVN, bge-reranker-v2-m3 raw, PhoRanker, Qwen3-Reranker-0.6B/4B zero-shot on dev_clean / dev_strict (Điều@1/3/5, MRR@10) — decides which base to fine-tune.
4. Fine-tune the winner(s) on the decontaminated data (+ synthetic queries, see retrieval-training-data).
Fair protocol: same candidate pool, Điều-level, dev_clean / dev_strict, paired tests + Holm, test once.
