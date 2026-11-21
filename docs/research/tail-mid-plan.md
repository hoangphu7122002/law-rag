# Plan: raising the tail and middle groups (decided 2026-11-22)

Context: dev300 mean .664. High (>=.8) 34 %, middle 48 % (54 % of the loss), tail 17 % (37 %). Null/negative so far: better reranking (fine-tuned, oracle, GLM/DeepSeek order +0), chunking Stage 1, cite weight, truncation, quote/MBR/selector. Only measured lever: more data (+.016 per doubling). SFT seed sd .015.

## Agreed (not yet run)
0. Step 0 diagnostics, no GPU: (a) dev300 ROUGE-L by number of gold articles (1 / 2 / 3+); (b) for the 21 wrong-law dev cases, check with the KG whether gold is in the same law family (amend/replace/consolidate) as the retrieved law, which is newer, and the effective dates of gold vs a common reference date. Stop the family idea if < ~8 cases.
1. Family expansion (only if step 0 passes): after rerank top-5, add the corresponding Dieu of same-family laws from the KG, rerank again, keep top-5/6. Never filter by validity (references often cite the older law: 9 of 14). Measure Dieu@5 with scripts/retr_compare.py first.
2. SFT data diversity: shuffle passage order for a fraction of rows, inject ALL gold articles (not one), controlled gold-absent rate (0 / 15 %). Configs A (current, 3 seeds exist), B (shuffle + inject-all), C (B + 15 % absent); 3 seeds each; stop early if mean B - A < .01 after 2 seeds. Implement test-first.
3. Teacher data (MANDATORY per user: teacher answers follow the reference template "Can cu theo Dieu X <Luat> ... nhu sau:" + verbatim quote, ~75 % of references) (DeepSeek-V4 flash / GLM-5.3 low via /vibe, no thinking needed): patterns single-article ~55 %, multi-article same law ~25 %, multi-law ~10 %, refusal 10-15 %. Source = corpus articles only, exclude the 569 gold articles of dev/test. Pilot 50 samples first: style fit (ROUGE-L of teacher answer vs real answer for train questions on the same articles). Check output licence terms before training.

## Direction (user, 2026-11-22)
- One-shot flow stays the deployed system (answering LLM < 4B), with the agreed enhancements.
- Add a ReAct flow with a big model (GLM-5.3 / DeepSeek-V4 via /vibe, or the 9B) that uses tools (search over our index, KG family lookup) to build correct context and answers; if the 9B (or the 2B) is wrong, the big teacher corrects it.
- Verified clean data (train questions: accepted only if it matches the reference / cites the gold Dieu; synthetic questions: grounded quote check) + the one-shot data tricks (template opener, shuffle, inject-all gold, absent rate, test-like context) -> train the 2B in the one-shot format.
- Gate: pilot first (50 train questions): (a) teacher ROUGE-L vs reference with gold context; (b) teacher-agent Dieu@5 vs the pipeline.

## Deferred / optional (not agreed, not about tail/mid)
- API: attach KG validity status to each source in a separate field (answer string unchanged, ROUGE-L unaffected); optional "valid laws only" retrieval mode, off by default, not scored. User: later.

## Extra ideas (mine, unmeasured)
- Classify the middle group by cause (wrong khoan/diem selected, answer too short/long, missing second article, style) with GLM low on dev labels; target the largest cause.
- Expected ceilings are small (+.007 to +.02 each); report as exploratory, 3 seeds before any claim.

## Not planned
Gateway/scaler, validity filtering in the scored pipeline, newer-law preference (refuted), hybrid retrieval (on par after rerank).
