# Zalo AI legal and Vietnamese legal RAG — what to borrow (2026-11-18)

Sources: `findings_zalo2021.md`, `findings_hf_models.md`, `findings_legal_rag.md` (same folder). Zalo 2021
top-3 placements and F2 scores were NOT found; evidence comes from public repos and papers on Zalo-style data.

## What is special in each line of work

| Work | Special part | Evidence |
|---|---|---|
| Zalo 2021 repo CuongNN218/zalo_ltr_2021 | BM25 → bi-encoders (PhoBERT-large, ViBERT, Condenser, Co-Condenser) with 2 rounds of hard negatives → weighted ensemble → score = cos × BM25 → keep articles within a margin of the top score (≤ 5) | repo README, no score stated |
| arXiv 2209.14494 (same pipeline, other corpus) | `sqrt(BM25) × cos` beats linear mixing; word segmentation (VnCoreNLP) helps PhoBERT; Condenser no clear gain; F2 .741 vs BM25+ .221 | paper |
| AITeamVN Embedding / Reranker | bge-m3 / bge-reranker-v2-m3 fine-tuned on 300k / 1.1M Vietnamese triplets; Zalo MRR@10 .818 → .867 with rerank (bge-m3 zero-shot .682) | model cards |
| BKAI bi-encoder (arXiv 2412.00657) | 7 bge-m3-mined hard negatives + InfoNCE; in-domain fine-tune doubles MRR on thuvienphapluat set (32.7 → 70.4) | paper |
| Vi-Qwen2-RAG | generator trained for noisy context, refusal, multi-doc; prompt `### Ngữ cảnh` (numbered) / `### Câu hỏi` / `### Trả lời`, temp 0.1 | model card, recipe unpublished |
| Graph legal RAG repos / papers | Chương → Điều tree, Neo4j KG, self-reflection loop; GraphRAG / HippoRAG / LightRAG compared only on multihop (ViHERMES) | snippet level |
| Thai NitiBench GRPO | reward = citation accuracy + BGE-M3 similarity, much cheaper than LLM judge | UNVERIFIED |

## What law-rag borrows (mapped to roadmap)

1. **Unit = Điều** (Zalo's retrieval unit is the article) → chunk `dieu` (phase 1 / step 2). Confirms the plan.
2. **Hybrid score `sqrt(BM25) × cos`**, BM25 on word-segmented text (pyvi / underthesea) → A4. Compare with RRF.
3. **Dynamic k**: keep chunks within a margin of the top rerank score, cap 5 → fewer noisy passages, shorter
   context; likely helps BLEU-1 (precision). Cheap, no re-embed → A4.
4. **Gold Điều from our own train answers**: answers start "Căn cứ khoản X Điều Y <law_id>", so parse
   (question, law_id, Điều) pairs → retrieval eval (`law_hit`, Điều hit) and training triplets for A7
   (MNRL / InfoNCE, hard negatives from the current model, skip top ranks; lr 2e-5, bs 64). Works where the
   law text exists (needs vbpl for most).
5. **Generator prompt**: keep Vi-Qwen2-RAG's own format (our `vi_rag` + `ngu_canh`), temperature ~0.1; add the
   "Căn cứ …" answer template (A3).
6. **GRPO reward** = metric mix (cos + ROUGE-L + BLEU-1) + citation correctness (law_id and Điều in the first
   sentence) + length penalty → A6 after SFT. No Vietnamese < 4B precedent found → novelty for the report.
7. **Graph**: no evidence that GraphRAG helps single-article questions; keep the lightweight graph from the
   corpus relation blocks (phase 4); HippoRAG / LightRAG only as one report ablation.

## Not worth copying
- Condenser / Co-Condenser pre-training, 4-model ensembles: costly, gain unproven, and AITeamVN models already
  start from a stronger base (bge-m3).
- Pseudo-labeling: hurt the Zalo ensemble.

## Gaps
- Official Zalo 2021 top-3 write-ups not found (portal not indexed); Vi-Qwen2-RAG training recipe unpublished.
