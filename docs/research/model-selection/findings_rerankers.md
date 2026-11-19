# Findings: rerankers with Vietnamese numbers (agent summary, 7 model cards fetched; (U) = unverified)
| Model | Size / max len | Vietnamese / multilingual numbers | Licence | FT tooling |
| AITeamVN/Vietnamese_Reranker | 0.6B (bge-m3 base) / 2304 (q 256 + p 2048) | Zalo legal 2021 (own card): Acc@1 .794, Acc@10 .974, MRR@10 .867 (bge-m3 bi-encoder .568 / .892 / .682) | Apache-2.0 | same arch as bge-v2-m3 → CrossEncoder / FlagEmbedding |
| itdainb/PhoRanker | 0.1B / 256 | MMARCO-vi dev: NDCG@10 .742, MRR@10 .683, NDCG@3 .663; 15 docs/s A100 fp16 (~3.3 s / 50) | Apache-2.0 | CrossEncoder (U) |
| BAAI/bge-reranker-v2-m3 | 0.6B / 512 in examples, 8192 base (U) | MMARCO-vi (ViRanker paper): NDCG@3 .609, NDCG@10 .687 | Apache-2.0 | FlagEmbedding torchrun + LoRA (r32, a64); CrossEncoder works (U) |
| Qwen3-Reranker 0.6B / 4B / 8B | 0.6 / 4 / 8B, 32K | MTEB-R 65.80 / 69.76 / 69.02; MMTEB-R 66.36 / 72.74 / 72.94; MLDR 67.28 / 69.97 / 70.19 (over Qwen3-Embedding-0.6B top-100); no Vietnamese-specific numbers | Apache-2.0 | LoRA/PEFT on a decoder; no official recipe; slower than encoders (U) |
| jina-reranker-v3 / v2 | 0.6B (131K) / 0.3B (1024 U) | MIRACL 66.83 / 63.65, MKQA 67.92 / 67.90 | CC BY-NC 4.0 | not documented |
| gte-multilingual-reranker-base | 306M / 8192 | chart only (arXiv 2407.19669) | Apache-2.0 | encoder cross-encoder; trust_remote_code (U); ~10x faster than decoder rerankers (card) |
| ViRanker (namdp-ptit, arXiv 2509.09131) | ~0.6B bge-m3 base (U) | MMARCO-vi NDCG@3 .682, NDCG@10 .730 (self-reported; MRR@10 beats PhoRanker) | not checked | not checked |
| mxbai-rerank-base / large-v2 | 0.5B / 1.5B (U) | not fetched; English-centric (U) | Apache-2.0 (U) | - |
| RankZephyr / RankGPT-style | ~7B | English only (U); sliding windows; seconds per query (U) | - | - |
Takeaways: no latency on the cards except PhoRanker (benchmark ourselves); Vietnamese_Reranker is the only one with Zalo-legal numbers (own card, in-domain); best published MMARCO-vi is PhoRanker .742 (base bge-v2-m3 .687); Qwen3-Reranker-4B best-supported multilingual option (8B adds little); no MIRACL-vi / VN-MTEB rerank numbers found. Sources: huggingface.co/AITeamVN/Vietnamese_Reranker, itdainb/PhoRanker, BAAI/bge-reranker-v2-m3, Qwen/Qwen3-Reranker-0.6B, jinaai/jina-reranker-v3, Alibaba-NLP/gte-multilingual-reranker-base, arxiv.org/pdf/2509.09131.
