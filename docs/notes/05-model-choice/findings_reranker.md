# Reranker findings (Vietnamese legal RAG) - 2026-11-17

Method: 7 web searches. No independent Vietnamese-legal reranker benchmark found (no ALQAC / BKAI / VN-MTEB-reranking rerank numbers for any model). All numbers below are SELF-REPORTED unless marked. "(unv)" = from my prior knowledge / aggregator, not verified on official card.

## Table

| Model | Params | Max len | License | Vietnamese numbers (self-reported) | Runtime support | Source |
|---|---|---|---|---|---|---|
| AITeamVN/Vietnamese_Reranker | ~568M (bge-reranker-v2-m3 ft, 1.1M VN triplets) | 8192 (unv) | Apache-2.0 (unv) | Legal Zalo 2021: Acc@1 .7944, @3 .9324, @5 .9537, @10 .9740, MRR@10 .8672 vs BGE-M3 dense alone .5682/.7728/.8382/.8921/.6822 => +0.185 MRR@10, +0.226 Acc@1. Card says Zalo not in training. No NDCG@10. | sentence-transformers CrossEncoder, FlagEmbedding, vLLM (XLM-R score, unv) | https://huggingface.co/AITeamVN/Vietnamese_Reranker |
| BAAI/bge-reranker-v2-m3 | 568M | 8192 | Apache-2.0 | MMARCO-VI (ViRanker paper table): NDCG@10 .6872, MRR@10 .6209. No legal number found. | FlagEmbedding, ST, vLLM | https://arxiv.org/pdf/2509.09131 |
| namdp-ptit/ViRanker | ~568M (bge-m3 + blockwise parallel transformer) | 8192 (unv) | Apache-2.0 (unv) | MMARCO-VI: MRR@10 .7107, NDCG@10 .7302 (paper self-reported). No legal number. | ST CrossEncoder | https://huggingface.co/namdp-ptit/ViRanker ; https://arxiv.org/pdf/2509.09131 |
| itdainb/PhoRanker | ~135M-class PhoBERT (unv); needs word-segmented VN input | 256 (unv) | Apache-2.0 (unv) | MMARCO-VI: MRR@10 .6830, NDCG@10 .7422. No legal number. Short max len = bad for Điều. | ST CrossEncoder | https://huggingface.co/itdainb/PhoRanker |
| BAAI/bge-reranker-v2-gemma | ~2.5B (Gemma-2B) | 8192 (unv) | Apache-2.0 per aggregators; family card (v2.5-gemma2) is "gemma" -> verify | none for VN | FlagEmbedding (LLM reranker), vLLM (unv) | https://bge-model.com/bge/bge_reranker_v2.html |
| Qwen3-Reranker-0.6B | 0.6B | 32K | Apache-2.0 | none VN-specific. MTEB-R 65.80 / MMTEB-R 66.36 (mirror; unverified vs official) | vLLM (score API w/ hf_overrides, unv), transformers; ST not native | https://huggingface.co/Qwen/Qwen3-Reranker-0.6B |
| Qwen3-Reranker-4B | 4B | 32K | Apache-2.0 | none VN. MTEB-R 69.76 / MMTEB-R 72.74 (mirror) | same | https://huggingface.co/Qwen/Qwen3-Reranker-4B |
| Qwen3-Reranker-8B | 8B | 32K | Apache-2.0 | none VN. MTEB-R 69.02 / MMTEB-R 72.94 (matches official card) | same | https://huggingface.co/Qwen/Qwen3-Reranker-8B |
| jinaai/jina-reranker-v3 | 597M listwise (Qwen3-0.6B base) | 131K ctx window (unv); 64 docs/pass | CC-BY-NC-4.0 (unv; v3.5 confirmed CC-BY-NC) -> non-commercial | BEIR 61.94, MIRACL 66.83 (no VN breakdown) | own `rerank()` API; not plain CrossEncoder | https://jina.ai/news/jina-reranker-v3-0-6b-listwise-reranker-for-sota-multilingual-retrieval/ |
| jina-reranker-v2-base-multilingual | 278M | 1024 (unv) | CC-BY-NC-4.0 (unv) | none VN | ST (trust_remote_code) | https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual |
| mxbai-rerank-base/large-v2 | 0.5B / 1.5B (Qwen2.5 base) | 8K (32K claimed, unv) | Apache-2.0 (aggregator, unv) | none VN; English/BEIR-focused | own mxbai-rerank pkg, vLLM (unv) | https://dev.co/ai/llms/mxbai-rerank-large-v2 |
| LLM-as-reranker (Qwen3-8B/Gemma pointwise yes/no, or listwise) | 4-8B | 32K+ | varies | no VN legal evidence found; slower, prompt-sensitive; Qwen3-Reranker is the tuned version of this idea | vLLM | n/a |

Not found: any Vietnamese-legal-specific reranker beyond Vietnamese_Reranker; any 2026 VN legal reranker (Qwen3-based) release. Related: arXiv 2609.27376 (VN labour law cross-lingual QA) has retrieval tables, rerank gain not extracted. VLSP 2025 DRiLL used bge-reranker-v2-m3 + hard-neg FT for legal retrieval (https://preview.aclanthology.org/anlp-naacl/2025.vlsp-1.20.pdf).

## Caveats
- Vietnamese_Reranker MRR .8672 on Zalo is vs dense BGE-M3 alone; your first stage is hybrid, so real gain smaller. Zalo is candidate-pool style; do not compare with VMTEB Zalo-legal numbers (embedder NDCG@10 .8650 / MRR@10 .8334 for AITeamVN/Vietnamese_Embedding) - different setup.
- MMARCO-VI is machine-translated, general domain; weak proxy for Điều/Khoản.
- Leak check: Vietnamese_Reranker training data (1.1M triplets) may overlap Zalo/legal sources; AGENTS.md test-leak rule -> eval on own held-out dev.

## Ranked shortlist
1. AITeamVN/Vietnamese_Reranker - only one with Vietnamese legal numbers (MRR@10 .867 Zalo), 568M, 8K ctx fits full Điều, ~tens of ms/pair on L40S.
2. Qwen3-Reranker-4B (0.6B as cheap baseline) - strongest multilingual generalist, Apache-2.0, 32K ctx; no VN legal numbers so must test; fits budget (100 docs x ~500 tok well < 1s on L40S with vLLM). Candidate for LoRA fine-tune on own legal data.
3. BAAI/bge-reranker-v2-m3 - baseline/ablation; ViRanker (MMARCO-VI MRR@10 .711) as alt if Vietnamese_Reranker fails.

Avoid: PhoRanker (short ctx), jina v2/v3 (non-commercial), bge-gemma (no gain evidence, license unclear).
Plan: run shortlist on own dev set, report MRR@10 / Acc@1 / Recall@5 with and without rerank, top-50 candidates from hybrid.
