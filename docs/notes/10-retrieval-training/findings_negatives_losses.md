# Findings: hard negatives + losses (bge-m3 / bge-reranker-v2-m3)

Source: 4 web searches (snippets, mostly secondary; unverified items flagged). Date 2026-11-19.

## 1. Hard negative mining

### Rank range
- BGE docs: "top-k shifted by N" -- skip top ~5, sample after. Warn top-K direct => false negatives.
  https://bge-model.com/tutorial/7_Finetuning/7.2.1.html (example is bge-base-en-v1.5, skip=5)
- BGE-M3 maintainer (HF discussion): in-batch negatives + BM25 or ANN negatives; repo has ANN mining script. No rank cutoff stated.
  https://huggingface.co/BAAI/bge-m3/discussions/32
- No source found confirming a "rank 10-100" window for bge-m3. Common practice (FlagEmbedding default, unverified): range 2-200, sample 7-15.
- GeoGPT-RAG: bge-m3 reranker trained w/ SimANS-mined negatives, >=16 negs/query. https://arxiv.org/pdf/2509.09686
- Korean-English cross-lingual bge-m3 study: followed NV-Retriever margin filter, abs 0.05 and relative 0.05 margins. https://arxiv.org/pdf/2507.08480
- Distillation study (LLM -> cross-encoder): up to 99 hard negs/query from first-stage. https://arxiv.org/html/2405.07920v1

### False-negative filtering (NV-Retriever, arXiv 2407.15831)
- Positive-aware mining: use positive's score as anchor to drop candidates too close to positive.
- TopK-PercPos: keep negative if sim(q,neg) <= margin_pct * sim(q,pos); margin_pct = 0.95 best (acc drops above 95%).
- TopK-MarginPos: keep if sim(q,neg) <= sim(q,pos) - abs_margin (e.g. 0.05).
- Teacher for mining scores: a strong embedding model (not the student).
- Reported (via secondary summary, unverified vs paper): LLM judge found naive top-k negs are false negatives 38.8% (NQ), 47% (StackExchange); positive-aware cut false negs 57% (NQ) / 50% (SE). PercPos 0.95 avg NDCG@10 0.5856 on their ablation. NV-Retriever-v1: 60.9 MTEB retrieval (BEIR), +0.65 over prior.
- https://arxiv.org/html/2407.15831v2 ; https://huggingface.co/papers/2407.15831 ; https://www.themoonlight.io/review/nv-retriever-improving-text-embedding-models-with-effective-hard-negative-mining ; https://particula.tech/blog/fine-tune-embedding-model-for-rag (blog; claims sentence-transformers `relative_margin`/`absolute_margin` params in mine_hard_negatives -- verify)

### Multi-round mining
- Not covered by searches. Known idea (ANCE-style, refresh index with current model) risks drift to false negatives and index cost; no numbers collected. Treat as unverified; do round 2 only with positive-aware filter + cross-encoder check.
- Law domain caveat: near-duplicate / amended articles (dedup, cross-refs) = high false-neg risk; filter with gold-set citation closure.

## 2. Bi-encoder losses
- InfoNCE / MultipleNegativesRankingLoss: in-batch + mined hard negatives, temperature ~0.02-0.05 (bge uses 0.02; unverified here).
- CachedMNRL / GradCache: huge batch w/o extra memory, slower training. Benefit = memory/batch size. No source reporting direct quality gain vs uncached. https://sbert.net/docs/cross_encoder/loss_overview.html (docs)
- Margin-MSE (Hofstatter et al.): student matches teacher margin (s(q,d+) - s(q,d-)) via MSE. MS MARCO DistilBERT bi-encoder w/ ensemble teacher: MRR@10 0.323 vs 0.299 no distillation. Survey: beats pointwise MSE; CE->BE two-step more effective than direct BE training. https://arxiv.org/pdf/2211.14876 ; https://arxiv.org/pdf/2010.06467
- Related: self-distillation with adaptive relevance margins https://www.alphaxiv.org/abs/2407.21515 (not read)
- Margin-MSE bonus: soft teacher scores partly absorb false-negative noise.

## 3. Reranker losses
- Pointwise BCE: each (q,d) binary. Listwise softmax: CE over group 1 pos + N negs (group size configurable; SWIFT default 7 negs). Memory scales per_device_bs * (1+N). https://swift.readthedocs.io/en/v4.4/BestPractices/Reranker.html
- Practitioner (single experiment): BCE + hard negs beat listwise/relative in-domain. https://engineered.at/articles/fine-tuning-a-reranker
- Academic (MS MARCO fine-tune): listwise > pairwise; for LLM distillation listwise gave no benefit over pairwise. https://arxiv.org/html/2405.07920v1
- LambdaLoss: no head-to-head found. Candidate to read: https://arxiv.org/pdf/2603.03010 (2026 controlled benchmark of cross-encoder training strategies; snippet only, results NOT read).
- bge-reranker-v2-m3 official FT uses listwise-style CE with group (1 pos + N neg), teacher-less.

## 4. Takeaways for law-rag
1. Mine from ANN/BM25 union, skip top ~3-5, take from ~top 5-100; apply PercPos 0.95 (or abs 0.05) vs positive score.
2. Use cross-encoder/LLM check on borderline negatives (legal near-duplicates).
3. Bi-encoder: MNRL (Cached if memory-bound) + hard negs; try Margin-MSE from bge-reranker-v2-m3 as ablation.
4. Reranker: A/B BCE vs listwise softmax (group 8-16); read 2603.03010 before LambdaLoss.
5. Gaps: multi-round mining numbers, GradCache quality gain, LambdaLoss -- no evidence collected.
