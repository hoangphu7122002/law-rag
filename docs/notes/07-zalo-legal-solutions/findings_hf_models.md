# Findings: Vietnamese legal retrieval models on HuggingFace

Collected by a research subagent. Scores are from model cards / papers and are NOT comparable across models
(different eval splits: AITeamVN full Zalo train, halong 20 % sample, BKAI own splits).

| Model | Base | Data | Max len | Zalo legal score | License |
|---|---|---|---|---|---|
| AITeamVN/Vietnamese_Embedding | bge-m3 | ~300k triplets | 2048 | Acc@1 .7274, Acc@10 .9568, MRR@10 .8181 (bge-m3 zero-shot: .5682 / – / .6822) | Apache-2.0 |
| AITeamVN/Vietnamese_Reranker | bge-reranker-v2-m3 | ~1.1M triplets ("not trained on Zalo", sources unnamed) | 2304 (q 256 + p 2048) | Acc@1 .7944, MRR@10 .8672 | Apache-2.0 |
| bkai vietnamese-bi-encoder (arXiv 2412.00657) | PhoBERT, 256-token chunks | synthetic pretrain → 80 % Zalo train + thuvienphapluat | 256 | MRR@10 79.31, R@10 68.02 (bge-m3 zs 64.43 / 57.28); TVPL set MRR@10 32.68 → 70.37 | – |
| hiieu/halong_embedding | multilingual-e5-base | ~100k in-house pairs, Matryoshka + MNRL | 512 | 20 % Zalo: NDCG@10 .8976, MRR@10 .8799, R@10 .9640 (768 d); NDCG .8743 @256 d | Apache-2.0 |

- BKAI recipe: InfoNCE, 7 bge-m3-mined hard negatives + in-batch, bs 64, lr 2e-5, ~5 epochs.
- DRiLL (VLSP 2025): bge-m3 dense → bge-reranker-v2-m3 fine-tuned with simple hard-negative mining.
  https://preview.aclanthology.org/anlp-naacl/2025.vlsp-1.20.pdf
- Unverified / not read: dangvantuan/vietnamese-embedding, truro7/vn-law-embedding, GreenNode, VN-MTEB, arXiv 2503.07470.

## Does fine-tuning bge-m3 on legal pairs help?
Likely yes (AITeamVN +16 pts Acc@1 over zero-shot; BKAI in-domain fine-tune large gains, on PhoBERT), but no clean
leak-free ablation found. Reranker adds +4.9 MRR@10 over the embedder (.8181 → .8672).

## Recipe (for roadmap A7)
- Triplets (question, gold Điều, hard negatives mined with the base model; skip top few ranks = false-negative risk).
- sentence-transformers MultipleNegativesRankingLoss (optionally MatryoshkaLoss) or FlagEmbedding finetune; lr 2e-5, bs 64.
- Reranker: fine-tune bge-reranker-v2-m3 on the same negatives.
- Eval on our own held-out split (AITeamVN weights may have seen Zalo data).

Sources: https://huggingface.co/AITeamVN/Vietnamese_Embedding · https://huggingface.co/AITeamVN/Vietnamese_Reranker ·
https://arxiv.org/html/2412.00657v1 · https://huggingface.co/hiieu/halong_embedding · https://huggingface.co/nrl-ai/vietnamese-embedding-bge-m3
