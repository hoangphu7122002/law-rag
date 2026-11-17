# Findings: embeddings for Vietnamese legal retrieval (2026-11-17)

Source: research subagent (web). SR = self-reported, IND = independent (VN-MTEB, arXiv 2507.21500).

| Model | Params | Max len | Dim | License | pyvi | Vietnamese numbers |
|---|---|---|---|---|---|---|
| AITeamVN/Vietnamese_Embedding (bge-m3 FT) | 568M | 2048 | 1024 | Apache-2.0 | no | Zalo legal Acc@1 0.7274, MRR@10 0.8181 (SR); VN-MTEB retrieval 34.18 (IND) < bge-m3 |
| Vietnamese_Embedding_v2 | 568M | 2048 | 1024 | Apache-2.0 | no | Acc@1 0.7262, MRR@10 0.8149 (SR) — not better than v1 |
| BAAI/bge-m3 | 568M | 8192 | 1024 + sparse + ColBERT | MIT | no | Zalo Acc@1 0.5682 (SR); VN-MTEB retrieval 39.84, 5/18 (IND) |
| multilingual-e5-large-instruct | 560M | 512 | 1024 | MIT | no | VN-MTEB retrieval 40.88 (IND) |
| multilingual-e5-large | 560M | 512 | 1024 | MIT | no | 37.65 (IND) |
| gte-multilingual-base | 305M | 8192 | 768 | Apache-2.0 | no | 38.38 (IND) |
| halong_embedding | 278M | 512 | 768 | Apache-2.0 | no | 34.45 (IND) |
| bkai vietnamese-bi-encoder | 135M | 256 | 768 | Apache-2.0 | yes | Zalo Acc@1 0.7109 (SR); VN-MTEB 25.37 (IND) |
| keepitreal/vietnamese-sbert (baseline) | ~135M | 256 | 768 | Apache-2.0 | yes | no legal numbers; PhoBERT family, likely weak |
| Qwen3-Embedding 0.6B / 4B / 8B | 0.6 / 4.02 / 8B | 32k | 1024/2560/4096 | Apache-2.0 | no | no Vietnamese numbers; Qwen-family gte-Qwen2-7B/1.5B top VN-MTEB retrieval 46.05 / 42.01 (IND) |
| jina v3 / v4 | 570M / 3.8B | 8k/32k | | non-commercial | | drop |
| EmbeddingGemma-300m | 300M | 2048 | 768 | Gemma | | no numbers, drop |

- Community legal FT: bqbbao6/vietnamese-legal-embedding (e5-base) Zalo NDCG@10 0.876 vs 0.603 base (SR, likely trained on Zalo train).
- Hybrid BM25 + dense: mixed. VLSP 2025 team: BM25+bge-m3+cross-encoder helped; arXiv 2609.27376: RRF worse than dense bge-m3; TELEN card: BM25 NDCG@10 0.725 vs bge-m3 0.545 on a vi legal set. → tune fusion weight on dev, don't assume.
- No ALQAC / BKAI-legal numbers found.

## Shortlist
1. AITeamVN/Vietnamese_Embedding v1 (+ Vietnamese_Reranker).
2. BAAI/bge-m3 (independent best easy model; sparse output = built-in hybrid; base for fine-tune).
3. Qwen3-Embedding-4B/8B (fits L40S; no vi legal evidence, must test). Note: embedder is not the answering LLM, 4B rule does not apply — confirm.
Also cheap: e5-large-instruct, gte-multilingual-base.

Sources: huggingface.co/AITeamVN/Vietnamese_Embedding · arxiv.org/html/2507.21500v1 · qwenlm.github.io/blog/qwen3-embedding ·
huggingface.co/bqbbao6/vietnamese-legal-embedding · huggingface.co/d4rkk3y/telen · preview.aclanthology.org/anlp-naacl/2025.vlsp-1.20.pdf ·
arxiv.org/pdf/2609.27376 · arxiv.org/html/2412.00657v1
