# Retrieval + reranker training to beat AITeamVN — data, models, protocol (2026-11-19)

Sources: findings_datasets_aiteamvn.md, findings_base_models.md, findings_recipes_eval.md (+ docs/notes/10-retrieval-training/). Mostly snippets: numbers unverified.

## What we are up against
- AITeamVN: Vietnamese_Embedding = bge-m3 fine-tune on ~300k triplets (v2: 1.1M); Vietnamese_Reranker = bge-reranker-v2-m3 fine-tune on ~1.1M triplets; recipes and data undisclosed; card says Zalo legal 2021 is in the training data (their Zalo scores are in-domain).
- Us: 1,452 decontaminated course questions + ~2.9k Zalo pairs (≈ 0.4 % of their reranker data). Already done: rr_v0 = their reranker fine-tuned on 4k groups: dev Điều@1 .587 → .712, val NDCG@10 .708 → .814, but NO generation gain (v3b + rr_v0: .6253 vs .6294 dev, .6101 vs .6073 test).
- Where the gap is: gold Điều is in top-5 for 90 % (Điều@20 .97) → recall is not the limit; ordering and, above all, which Điều the LLM cites among the 5 are (cites another Điều: rougeL .397 vs .702).

## Data inventory (decontaminate everything vs test: Điều pairs + 13-grams)
| Source | Size | Use | Risk / note |
|---|---|---|---|
| Course train (decontam) | 1,452 q | positives + hard negatives (dense top-50) | already built (rerank_data.py) |
| Zalo Legal 2021 | ~2.9k q, 61k articles | positives; negatives same law | already used; AITeamVN also trained on it → do not evaluate on Zalo |
| LLM synthetic queries per Điều (Haiku 5.5 low, paced; filter: no passage copy, round-trip hit in top-20, dedupe) | target 20–50k | the scale gap; S1 of notes/10 | decontam vs test + dev; teacher output never in dev/test |
| BKAI synthetic (arXiv 2412.00657) | ~500k pairs (reported) | pretrain stage | public release unconfirmed; check HF bkai-foundation-models/crosslingual |
| ALQAC / VLQA / VLSP | small / >3k | extra positives | registration; ALQAC 2025 bars external annotated sets; release unverified |
| Reranker-teacher listwise labels (our rr_v0 / Qwen3-Reranker-4B on top-50 of train + synthetic questions) | any | distillation targets (Rank-DistiLLM: ~10k queries enough, pairwise loss fine) | cheap, no new annotation |

## Base models (retriever size is unconstrained; 2 × 46 GB)
1. Keep: AITeamVN/Vietnamese_Embedding and Vietnamese_Reranker (fine-tune them: already shown to move Điều@1 +.125).
2. Add as challengers (hypotheses, no Vietnamese-legal numbers): Qwen3-Reranker-4B (MMTEB-R 72.7, 32K ctx, Apache-2.0, LoRA ~30–40 GB) and Qwen3-Embedding-4B (MMTEB 69.5, LoRA); bge-reranker-v2-m3 raw as the reranker control.
3. Skip: jina v3 / v4 / reranker (non-commercial); multilingual-e5 (512-token limit, ~+1 point on VN-MTEB retrieval only).

## Recipe (in order of expected gain per hour)
1. **Target = what the generator needs, not only "the gold Điều is ranked first".** Cheap test first (no training): rr_v0 with k = 1–3 in generation on dev_clean. If rr_v0's Điều@1 .712 only pays off when fewer, better passages are shown, that is the lever.
2. Reranker: fine-tune with hard negatives that look like the generation failure (same-law neighbours, amended/replaced versions via the KG), semi-hard rather than hardest, false-negative filter at 95 % of the positive score (already in rerank_data.py), pointwise/pairwise loss, then distil an LLM reranker's order.
3. Embedder: MNRL with 7 hard negatives, temperature .02, lr 1e-5, only if Điều@50 becomes the limit (today Điều@20 .97).
4. Forgetting: interpolate base ↔ fine-tuned (α .5–.8, unverified), keep the base row in every table.

## Fair evaluation vs AITeamVN
- Same candidate pool (our corpus, same chunks, same top-50 first stage), same Điều-level aggregation; metrics Điều@1/3/5, MRR@10, nDCG@10; on dev_clean / dev_strict (no Điều / 13-gram shared with train); NOT on Zalo.
- Rows: AITeamVN as-is · raw bge-reranker-v2-m3 · rr_v0 · new variants; per-query paired tests with CIs, Holm correction over the variants; several seeds for the best one; test once at the end (test-informed note).
- Report generation too (rougeL) since retrieval gains have not transferred so far.

## Risks
- Zalo-trained AITeamVN looks strong on Zalo for in-domain reasons; our dev is course-style.
- Synthetic queries are lexically close to their Điều → inflated dev gain; the strict dev subset guards against it.
- Haiku use: paced batches, decontaminated output (memory: haiku pacing).
