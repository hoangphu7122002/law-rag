# Model choice for law-rag — report (2026-11-17)

Inputs: findings_llm.md, findings_embedding.md, findings_reranker.md. Param counts verified via HF API `safetensors.total`.

## Verified parameter counts (answering LLM must be < 4B)

| Model | Params | License | Verdict |
|---|---|---|---|
| Qwen3.5-0.8B | 0.873B | Apache-2.0 | OK (small) |
| Qwen3-1.7B | 2.032B | Apache-2.0 | OK |
| Qwen3.5-2B | 2.274B (incl. vision tower) | Apache-2.0 | OK |
| Qwen2.5-3B-Instruct | 3.086B | Qwen Research | OK |
| AITeamVN/Vi-Qwen2-3B-RAG | 3.086B | Apache-2.0 | OK |
| arcee-ai/Arcee-VyLinh | 3.397B | Apache-2.0 | OK |
| Qwen3.5-4B | 4.660B | | **excluded** |
| Gemma 4 E2B | 5.1B total | | **excluded** |

## Decision proposal

| Component | Default | Ablate |
|---|---|---|
| LLM | Qwen2.5-3B-Instruct (stable vLLM) | Qwen3.5-2B, Vi-Qwen2-3B-RAG (zero-shot, context-only RAG training), Arcee-VyLinh, Qwen3-1.7B |
| Embedding | AITeamVN/Vietnamese_Embedding (cached) | bge-m3 dense+sparse (cached), Qwen3-Embedding-4B, e5-large-instruct |
| Lexical | BM25 (pyvi) — fusion weight tuned on dev | |
| Reranker | AITeamVN/Vietnamese_Reranker | bge-reranker-v2-m3, Qwen3-Reranker-4B |
| Teacher (offline) | Qwen3.5-9B (already on L40S) | |

## Gaps
- All Vietnamese legal numbers are self-reported (AITeamVN cards); independent VN-MTEB ranks Vietnamese_Embedding below bge-m3.
- No faithfulness / legal QA numbers for any < 4B LLM → dev-300 decides.
- Qwen3.5-2B needs a recent vLLM; check vllm_env 0.30 support.
