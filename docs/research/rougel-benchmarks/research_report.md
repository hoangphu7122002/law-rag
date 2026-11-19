# Is ROUGE-L 0.8–0.9 reported for Vietnamese legal RAG / QA? (2026-11-20)

Sources: findings_published_vn.md, findings_code_solutions.md, findings_rouge_levels.md (web search; "verified" = number read in the fetched source, otherwise UNVERIFIED snippet).

## Verdict
No verified system reports ROUGE-L 0.8–0.9 for free-form legal QA scored against independent references. The best comparable numbers are ~0.66 (style-matched fine-tuning) and 0.756 (one weak, unreproducible paper). Our dev ROUGE-L 0.66 (C1) is at the top of what is published for reference answers of this kind; 0.8–0.9 is not a credible target without changing what is being scored.

## Numbers
| System / setting | ROUGE-L | Variant | Source | Status |
|---|---|---|---|---|
| Vietnamese legal opinion / judicial reasoning / summarisation, zero-shot LLMs (VLegal-Bench) | 0.50 / 0.42 / 0.33 best | unspecified | arxiv.org/html/2512.14554v4 | verified |
| VLegal-Bench task 2.3 Legal Graph Structuring (not QA, structured output) | 0.808 (Claude Sonnet 4.5, Qwen2.5-72B) | unspecified | same | verified |
| VLQA statute QA, LLMs (GPT-4o-mini 2-shot best) | 0.38–0.62 | | | verified (per agent) |
| VLQA fine-tuned ViT5 / BARTpho, free-form | 0.32–0.33 | | | verified |
| ViBidLAQA (bidding law, ViT5 fine-tuned on same-domain LLM-generated refs) | 0.657 (BLEU-4 .395, BERTScore .867) | | HF model card | UNVERIFIED |
| Hien, Nhi, Huyen 2025 (GPT-4 + Milvus, 2,000 pairs, "expert-validated" + template-generated refs) | 0.756 (ROUGE-1 .834) | F/P/R not stated | scitepress.org/publishedPapers/2025/137512 | verified read, rated weak (no code, ref construction unclear) |
| Chinese statute QA, zero-shot LLMs | 0.22–0.27 | | | UNVERIFIED |
| Our C1 (dev300, 2B SFT v3b + 2 shots + MBR8) | 0.658 | F1, course metric | this repo | measured |

Not found: ROUGE-L for ALQAC, Zalo, VLSP, ViLegalLM, URA-LLaMa, ViLQA, AITeamVN models (they report retrieval metrics / accuracy); no GitHub/Kaggle report for this assignment.

## What would make 0.8–0.9 possible (reasoning, no source confirms it)
- reference answers that quote the retrieved article near-verbatim in a fixed template AND a system that copies the right passage with matched length; or a lenient scorer (recall instead of F1, ROUGE-Lsum, char/syllable level, heavy normalisation).
- SFT on the same-style references is the largest lever (format, length, phrasing), already used (v0/v3b).
Our measured ceiling: see oracle run (gold Điều injected) in docs/CHANGELOG.md; with the right Điều the average ROUGE-L was ~0.70 in the earlier error analysis.

## Consequence for the project
Target 0.66–0.70 on this metric; report all five metrics (cosine is already 0.91), paired bootstrap, baseline (~0.17) and the oracle ceiling. Follow-ups not done: ViLegalLM PDF, VLSP 2025 tables.
