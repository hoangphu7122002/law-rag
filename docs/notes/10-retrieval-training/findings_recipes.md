# Fine-tuning recipes: bge-m3 + bge-reranker-v2-m3 (VN legal)

Status: web search (4 queries) + prior knowledge. Items marked [mem] are from model memory, NOT verified; check before relying.

## 1. FlagEmbedding embedder (bge-m3) hyperparameters
- Official README: lr 1e-5 / 2e-5 / 3e-5 for large / base / small. bge-m3 = large -> start 1e-5.
- `train_group_size` = 1 positive + (N-1) negatives. #neg = group_size - 1.
- Seen in community bge-m3 runs (HF discussions, unverified quality):
  - lr 1e-5, temperature 0.02, group_size 2, custom data.
  - lr 5e-6, temp 0.01, group_size 6, 4 epochs.
  - lr 1e-5, temp 0.02, group_size 6, 5 epochs.
- LM-Cocktail paper (BGE finetune via FlagEmbedding): lr 2e-5, temp 0.02, AdamW, batch 256.
- [mem] FlagEmbedding example: bge-m3 finetune uses `--query_max_len 512 --passage_max_len 512` (raise to 1024 for long legal Dieu), `--warmup_ratio 0.1`, in-batch negatives + `--negatives_cross_device`, fp16/bf16, gradient checkpointing, deepspeed zero0/1. Large per-device batch matters (in-batch negs): use grad-checkpoint + cross-device negs to reach effective batch 64-256.
- [mem] Hard negative mining in FlagEmbedding: `hn_mine.py`, sample from rank range 10-210 (skip top-10 to avoid false negatives), `--negative_number 15`. Query-side prefix none for bge-m3.
- Knowledge distillation option: FlagEmbedding docs use bge-reranker-v2-m3 as teacher scorer on mined hard negatives (pass `pos_scores`/`neg_scores`, `--knowledge_distillation True`).
- Third-party sentence-transformers fine-tunes of AITeamVN embedding (ALQAC): MultipleNegativesRankingLoss, scale 20, per-device batch 16 x grad-accum 4, 10 epochs, warmup 100 (HF check2207/alqac_VNTeamVN_embedding). Another bge-m3 tune: lr 2e-5, eff. batch 32 on 2 GPUs.
- Practical small-data default for us: lr 1e-5, temp 0.02, group_size 8 (1 pos + 7 hard neg), batch >=32 effective, 1-3 epochs, warmup 10%, max len 512-1024, eval on dev each epoch, early stop.

## 2. FlagEmbedding reranker (bge-reranker-v2-m3)
- Search did not surface an official v2-m3 finetune recipe. [mem] FlagEmbedding `reranker` finetune: `--learning_rate 6e-5` for base-sized, `--train_group_size 16`, `--query_max_len 256 --passage_max_len 256`, `--num_train_epochs 1-2`, `--per_device_train_batch_size 2-4`, `--gradient_accumulation_steps`, listwise CE loss over group (1 pos + 15 neg). For a 568M XLM-R-large backbone use a lower lr (1e-5 to 2e-5) [mem, judgment].
- Data format same as embedder: `{query, pos:[..], neg:[..]}`; negatives = hard negs from the first-stage retriever (top 10-100 minus gold).
- Rerank pitfalls: false negatives (unlabeled relevant Dieu) hurt; mine from ranks beyond the gold neighborhood or filter with the current reranker score.
- AITeamVN Vietnamese_Reranker: bge-reranker-v2-m3 finetuned on ~1.1M triplets (Reranker/Embedding_v2 card). Hyperparameters not on the card.

## 3. How much data
- AITeamVN Vietnamese_Embedding: fine-tuned bge-m3, ~300k (query, pos, neg) triplets; max seq 2048, dim 1024, dot-product. v2 + reranker: ~1.1M triplets. Negative mining method not documented.
- BKAI (arXiv 2412.00657, "Improving Vietnamese Legal Document Retrieval using Synthetic Data"): 500k synthetic legal queries generated with Llama 3 from passages; pretrain bi-encoder + ColBERT on synthetic, then fine-tune with contrastive loss + mined hard negatives. Paper is bi-encoder/ColBERT; cross-encoder only discussed as background. BKAI legal bi-encoder on HF: bkai-foundation-models.
- Related VN legal pipeline (arXiv 2507.14619): bi-encoder + cross-encoder, MRR@10 79.11 dev / 77.54 private test.
- [mem] Rule of thumb: 1-5k queries with good hard negatives gives gains for domain adaptation (typically few pts recall@k); beyond that, add synthetic LLM queries (BKAI path) for 10-100k. With <5k, use low lr, 1-2 epochs, heavy regularisation/merging.

## 4. Avoid catastrophic forgetting
- LM-Cocktail (arXiv 2311.13534; pip `LM-Cocktail`): weighted average of finetuned + base (+ peers). BGE numbers (bge-base-en, 9 tasks): base 71.8 target / 49.8 other; plain finetune 76.0 / 48.5; merge 2 models 74.8 / 50.0; merge 10 models 74.7 / 50.6. => merging costs ~1 pt target, recovers/improves general. Works with no extra data.
- Recipe: `mix_models(model_names_or_paths=[ft, base], model_type='encoder', weights=[0.7..0.5, rest])`. Sweep alpha on dev; our target is VN legal so weight ft higher (0.7-0.9) [judgment].
- LoRA vs full: LoRA/low-rank limits drift; search found no bge-m3 LoRA numbers. [mem] LoRA r=16-64 on all linear layers, lr 1e-4 ok; full FT w/ lr 1e-5 + merge is the FlagEmbedding-supported path. Not verified for bge-m3.
- Mixing general data: replay a slice of general/MS MARCO-like or mMARCO-vi pairs (10-30%) [mem]. Keep legal dev + a general VN retrieval held-out to detect drift.
- Also keep bge-m3 sparse/colbert heads: unified finetune in FlagEmbedding (`--unified_finetuning True`) trains dense+sparse+multivec together; dense-only finetune may degrade sparse/colbert fusion.
- Tooling note: `use_reentrant` gradient-checkpointing issue in bge-m3 HF discussion #10; set `gradient_checkpointing_kwargs={"use_reentrant":False}`.

## 5. Vietnamese competition leads (not fetched in detail; todo)
- Zalo AI 2021 Legal Text Retrieval: [mem] winning pipelines = BM25 (top-k) + fine-tuned PhoBERT/XLM-R cross-encoder rerank, with negatives from BM25 top ranks, plus score ensemble. Needs verification.
- ALQAC 2022/2023 (VN): [mem] BM25 + multilingual-E5/bge/PhoBERT finetune, cross-encoder rerank, LLM rerank; ensembles of lexical+dense. HF `check2207/alqac_VNTeamVN_embedding` is an ST finetune of AITeamVN embedding for ALQAC.
- Search budget (4) exhausted; fetch 2507.14619 and 2412.00657 HTML for exact hyperparameters next.

## Sources
- https://huggingface.co/AITeamVN/Vietnamese_Embedding
- https://huggingface.co/AITeamVN/Vietnamese_Reranker
- https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2
- https://huggingface.co/check2207/alqac_VNTeamVN_embedding
- https://arxiv.org/abs/2412.00657 (html: https://arxiv.org/html/2412.00657v1)
- https://arxiv.org/html/2507.14619v1
- https://arxiv.org/pdf/2311.13534 (LM-Cocktail); https://pypi.org/project/LM-Cocktail
- https://github.com/FlagOpen/FlagEmbedding (examples/finetune/embedder, reranker)
- https://huggingface.co/BAAI/bge-m3/discussions/10
