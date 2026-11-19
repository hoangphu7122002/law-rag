# Findings: embedders with Vietnamese numbers (agent summary; VN-MTEB Table 3 verified, rest (U))
VN-MTEB (arXiv 2507.21500 Table 3; retrieval = mean nDCG@10 % over 15 translated-English datasets; per-dataset scores in Appendix H, not read):
| Model | Size | Retrieval |
| gte-Qwen2-7B-instruct | 7B | 46.05 |
| gte-Qwen2-1.5B-instruct | 1.5B | 42.01 |
| e5-Mistral-7B-instruct | 7B | 41.73 |
| multilingual-e5-large-instruct | 560M | 40.88 (best sub-1B) |
| bge-m3 | 568M | 39.84 |
| gte-multilingual-base | 305M | 38.38 |
| multilingual-e5-large | 560M | 37.65 |
| KaLM-embedding-m-mini-v1 | 494M | 35.07 |
| halong-embedding | 278M | 34.45 |
| AITeamVN/Vietnamese_Embedding | 568M | 34.18 (BELOW bge-m3 on general retrieval) |
| bkai vietnamese-bi-encoder | 135M | 25.37 |
| bge-multilingual-Gemma2 | 9B | 20.52 (anomalous) |
Paper: RoPE models beat absolute-position models.
Zalo legal 2021 (self-reported model cards, in-domain for the legal fine-tunes): MRR@10 / Acc@10 — Vietnamese_Reranker .867 / .974; Vietnamese_Embedding .818 / .957; Embedding_v2 .815 / .958; bkai bi-encoder .795 / .930; bge-m3 .682 / .892. Legal fine-tuning gives a clear Zalo gain (bge-m3 .682 → .818) yet the same model is worse than bge-m3 on general VN-MTEB → the gain is domain-specific (and partly inflated: Zalo is in its training data).
Not found: Qwen3-Embedding Vietnamese scores (only MTEB-Multilingual 70.58 for 8B, arXiv 2506.05176; card lists Vietnamese), MIRACL-vi, jina-v3, Arctic-embed 2, NV-Embed-v2, OpenAI / Gemini embeddings, GreenNode, ViLegalBench / VLegal-Bench.
Long inputs / prefixes (model cards + own knowledge, U): Vietnamese_Embedding 2048 tokens; bge-m3 and gte-multilingual 8192; Qwen3-Embedding 32K; multilingual-e5 512; bkai bi-encoder 256 (PhoBERT) → 500–3000-token articles need bge-m3 / gte / Qwen3 or chunking. Prefixes: e5 `query:` / `passage:`; e5-instruct and Qwen3-Embedding instruction on the query only; bge-m3 and Vietnamese_Embedding none. Sparse + ColBERT outputs only in bge-m3 itself (its fine-tunes keep the dense head).
