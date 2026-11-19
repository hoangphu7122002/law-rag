# Findings — synthetic queries for retriever / reranker training (2026-11-19, 4 searches, snippets; [memory] = unverified)

## Methods
- **InPars-v2** (https://arxiv.org/abs/2301.01820v3): 100k docs per BEIR set, 1 query/doc (GPT-J, 3 MS MARCO shots),
  monoT5 picks the top pairs; ~30 h on one A100; BM25 + monoT5 → BEIR SOTA.
- **Promptagator** (https://arxiv.org/pdf/2209.11755): 2–8 real in-domain shots per task prompt; dense retriever
  47.8 avg nDCG@10 on 11 BEIR sets vs ColBERTv2 46.2; round-trip filter (keep q only if a retriever trained on the
  synthetic data retrieves the source) — authors admit circularity.
- **GPL** [memory]: mine hard negatives, cross-encoder soft-labels (q, pos, neg), MarginMSE → robust to false negatives.
- **doc2query / GenQ** [memory]: 1–5 queries per passage.

## Filtering
- Round-trip: cheap, effective, drops a lot, biased to easy lexical queries; SQUARE (2025) calls it a weak error filter
  when a query matches several docs. CONVERSER ablation: filtering ≈ +0.1 → setup-dependent (https://arxiv.org/pdf/2309.06748).
- Alternatives: reranker score (InPars-v2, GPL), LLM relevance check (https://arxiv.org/pdf/2311.07930).

## Queries per doc
No tuned optimum found; common: sample 2–5 per doc, keep 1–2 after filtering, cover every doc.

## Prompt tips
Real in-domain few-shot examples, rotated across calls; a short "why relevant" field [memory]; tell the model not to copy
passage phrases; match real query length / style.

## Pitfalls
Lexical-copy queries; false negatives from near-duplicate articles; dev/test leakage via few-shot or generation →
decontaminate; round-trip circularity; weak generators can hurt (TAS-B on T5 queries mixed vs base,
https://arxiv.org/pdf/2502.19712v1).

## Vietnamese legal
BKAI "Improving Vietnamese Legal Document Retrieval using Synthetic Data" (https://arxiv.org/abs/2412.00657):
~500k LLM queries, pre-train bi-encoder + ColBERT, then contrastive fine-tune on mined hard negatives; gains on TVPL and
Zalo 21 (no independent replication found).

## Suggested (inference)
4–8 real train shots, 3 queries per Điều at T ≈ 0.8 → round-trip + cross-encoder filter, drop high-overlap copies →
mine negatives, relabel with reranker → decontaminate vs dev + test; ablate filter on/off and queries per doc.
Other: InPars toolkit https://arxiv.org/abs/2307.04601, round-trip QA https://arxiv.org/pdf/2002.09599.
