# Embedder benchmark (Modal, 2026-11-19)

Pool: 9,666 passages = AITeamVN top-50 candidates of 300 dev questions + gold; 264 questions with a gold Điều, 260 gold in pool.
Metrics: Điều@k = gold Điều among top-k passages (by Điều), MRR; throughput = passages/s encoding, latency = p50 per query (ms). L40S, fp16.
dev_clean n=130, dev_strict n=61 (subsets without test-contaminated rows).

| model | params | Điều@1 | @3 | @5 | @20 | MRR | clean @1 / MRR | strict @1 / MRR | pass/s | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| **aiteamvn** (current) | 0.57B | .534 | .777 | .871 | .955 | .674 | .569 / .695 | .557 / .683 | 159 | 14.7 |
| qwen3-emb-4b | 4B | .508 | .761 | .841 | .932 | .647 | .554 / .669 | .541 / .647 | 18 | 41.9 |
| qwen3-emb-8b | 8B | .511 | .758 | .830 | .928 | .653 | .462 / .613 | .377 / .561 | 13 | 41.6 |
| bge-m3 | 0.57B | .523 | .701 | .799 | .909 | .642 | .538 / .641 | .508 / .607 | 175 | 15.7 |
| e5-large-instruct | 0.56B | .477 | .716 | .780 | .924 | .617 | .446 / .593 | .426 / .567 | 197 | 14.6 |
| qwen3-emb-0.6b | 0.6B | .428 | .693 | .792 | .898 | .581 | .431 / .571 | .426 / .562 | 60 | 32.7 |
| gte-multilingual | 0.3B | .439 | .652 | .739 | .905 | .574 | .454 / .575 | .410 / .523 | 304 | 9.0 |
| gte-qwen2-7b | 7B | error (DynamicCache.get_usable_length: remote code vs transformers>=4.51) | | | | | | | | |

bge-m3 first failed (torch 2.5.1 vs torch.load safety check), re-run alone with torch 2.6.0 (separate app).

## Reading
- **No model beats AITeamVN on this pool.** But the pool is AITeamVN's own top-50, so it is biased in its favour; every other model is also capped by that candidate set. This measures re-scoring ability inside the pool, not full-corpus recall.
- Big models (Qwen3 4B/8B) are close (.51 vs .53 @1) at 9-12x lower throughput and ~3x query latency: not worth it on this evidence. 8B is worse than 4B on dev_clean/strict (small n, noisy).
- bge-m3 is the closest small model (-0.01 @1, similar speed); useful as a second dense retriever for fusion, not a replacement.
- Small multilingual-e5 / gte are clearly behind; Qwen3-0.6B is slow and weaker.
- Next (if pursued): fine-tune AITeamVN on synthetic Haiku queries (docs/research/retrieval-training-data) and judge on the full corpus; fuse AITeamVN + bge-m3 + BM25.
- gte-qwen2-7b not measured (needs transformers<4.48 image); skipped, 4B/8B Qwen3 already show the large-model trend.

Cost: Modal L40S, profile phuchuynhvinh-fwork, ~35 min main + ~5 min bge re-run => about $1.3.
