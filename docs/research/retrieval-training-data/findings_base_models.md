# Findings: base models (agent summary; (U) = unverified)
Baseline AITeamVN/Vietnamese_Embedding: bge-m3 fine-tune 568M, Zalo legal MRR@10 .8181 (raw bge-m3 .6822), Reranker MRR@10 .8672 — in-domain caveat.
| Model | Size / ctx | Scores | Licence | FT |
| Qwen3-Embedding 0.6B / 4B / 8B | 0.6 / 4 / 8B, 32K | MMTEB mean 64.33 / 69.45 / 70.58 | Apache-2.0 | ST ≥2.7, LoRA; 4B LoRA ~30-40 GB, 8B tight on 2 GPUs (U) |
| multilingual-e5-large-instruct | 560M, 512 | VN-MTEB retrieval 40.88, avg 67.99 | MIT (U) | ST, ~16 GB; 512-token limit is bad for articles |
| gte-multilingual-base | 305M, 8192 | VN-MTEB retrieval 38.38 | Apache-2.0 (U) | ST <12 GB |
| bge-m3 (ref.) | 568M, 8192 | VN-MTEB retrieval 39.84, avg 64.90 | MIT | ST / FlagEmbedding ~16-20 GB |
| jina-embeddings-v3 / v4 / reranker-v3 / v2 | 570M / 3.8B / ~597M | none found | CC-BY-NC / Qwen Research (non-commercial) | deprioritise |
| Arctic-embed 2.0-l | ~568M (U) | none found | Apache-2.0 (U) | ST ~16 GB |
| Qwen3-Reranker 0.6B / 4B / 8B | 0.6 / 4 / 8B, 32K | MMTEB-R 66.36 / 72.74 / 72.94 | Apache-2.0 (U) | LoRA via HF/PEFT (no ST CrossEncoder recipe); 4B LoRA ~30-40 GB |
| bge-reranker-v2-m3 | 568M, 8192 | not fetched | Apache-2.0 (U) | ST CrossEncoder, ~16-20 GB |
No Vietnamese-legal or Zalo scores for any candidate; VN-MTEB (arXiv 2507.21500) read only partly. Qwen3-Emb-4B and Qwen3-Reranker-4B look strongest on paper (long context, Apache-2.0, LoRA): a hypothesis until tested on Zalo / our dev. Sources: huggingface.co/Qwen/Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, arxiv.org/html/2507.21500v1, arxiv.org/html/2506.05176v3, jina.ai/en-US/models/.
