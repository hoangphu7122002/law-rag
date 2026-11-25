# Tail / mid log (night of 2026-11-25 → 26). Metric = `rouge_l` column of data/runs/results.tsv (NOT jaccard). Each run's model verified via config.yaml generate.model.

## Correction (2026-11-26 00:10)
Earlier chat numbers for D3c used the jaccard column by mistake, and "C1" was actually run q35_D_pissa_last (old D). Correct ROUGE-L, dev300 / test200:
| model | dev | test |
|---|---|---|
| C PiSSA (V_pissa, data C) | .6414 | .6307 |
| C rsLoRA | .6319 | .6165 |
| D3c PiSSA last | .6320 | .6252 |
| D3c rsLoRA best-val / last | .6258 / .6258 | .6233 / .6201 |
| D3c PiSSA best-val | .6225 | .5987 |
| dpo_a2 (from v3b, older pipeline, CHANGELOG) | .6637 | .6433 |
Stages on dev (n / rouge_l), C PiSSA vs D3c rsLoRA: gold not in passages 25 / .323 vs .331; ref cites no article 36 / .574 vs .521; retrieved but cites another Điều 24 / .408 vs 31 / .413; retrieved and cited 211 / .724 vs 206 / .714. Refusal rate 1.0% vs 0.7%. => D3 data does not beat C on ROUGE-L; weak spot common to all models = "retrieved but cites another Điều".

## Research agent conclusions (sonnet55-med-tailmid-research, search-summary level evidence, gains are estimates)
- Group a (gold in context, model quotes another plausible article; ~10% of dev, mean .41): RAFT-style neighbour-distractor SFT (gold + Điều N±1..3 of the same law + same-concept article of another law; KG builds contexts, no LLM); Trust-Align-style DPO pairs (chosen = gold-citing answer, rejected = on-policy answer citing the plausible wrong article, length matched). Estimated +.005..+.015 overall.
- Group b (gold missing, 25 dev): do not add refusals (refusal ~0 ROUGE-L); keep teacher refusal <=5%.
- Group c (reference cites no article, 36 dev): only train a no-cite style if a predictable question cue exists; classify the 36 first; cap <=10%.
- Free diagnostics first: answer length ratio by group, ROUGE by number of gold articles, wrong-cite counts; length-matched DPO if a length bias shows. MBR/selector previously null in this project.
- Haiku designs (later, slowly): A sibling contrast (question only the gold answers; second call checks the distractor does not answer), B minimal pairs (two questions on the same entity needing different articles), C no-cite style (only if step 0 finds a pattern). Common filters: train-split articles only, drop dev/test gold articles, question-overlap filter vs dev/test, verify every Điều and number against the article text, verbatim quote check, cited Điều == gold Điều.
- Ranked: 1) step-0 diagnostics 2) neighbour-distractor SFT (config beside A/B/C) 3) wrong-article on-policy DPO pairs 4) Haiku Design B pilot (~300 rows) 5) length-calibrated DPO reward if a length bias exists.

## CPT v2 data (2026-11-26 00:31)
data/cpt/mix.npy = 102,132 sequences x 2048 = 209.2M tokens (real tokenizer): raw laws 8.0M, DeepSeek augmentation 6.7M, external law 174.8M (th1nhng0 55M est., tmquan 35M, court judgments 30M PII-masked, Pháp điển 19.8M, án lệ 12M, UTS 6M, Wikisource 3M), general replay 11.4M (FineWeb-2 vie, Wikipedia vi), SFT D3 replay 8.0M. All chunks decontaminated (13-gram) vs dev/test questions and references. CPT v2 starts when both GPUs are free; SFT-from-CPT queues (PiSSA on data C and on D3) start automatically after cpt_report.json.

## DPO round 1 on D3c rsLoRA best-val (dpo_D3c_a), dev300 — 2026-11-26 00:48
rouge_l dev **.6195** vs base D3c rsLoRA .6258 (-.0063, inside seed noise ~.015 but NOT an improvement). Refusal 0.7% (same), len ratio 1.03 (base 1.02). Stages (n / rouge_l), DPO vs base: gold not in passages 25 / .340 vs .331; ref cites no article 36 / .510 vs .521; retrieved cites other Điều 27 / .425 vs 31 / .413; cites none 2 / .358 vs .390; retrieved and cited 210 / .699 vs 206 / .714. => DPO helps the weak groups slightly (+.01) but loses on the large "retrieved and cited" group (-.015). Pairs: 1347 (margin .39, chosen 1585 vs rejected 1931 chars). Test eval running next; PiSSA DPO (dpo_D3cP_a) train finishing ~00:52.

## DPO round 1 on D3c rsLoRA best-val (dpo_D3c_a), test200 — 2026-11-26 00:53
rouge_l test **.6016** vs base D3c rsLoRA .6233 (**-.022**); dev .6195 vs .6258 (-.006). DPO round 1 from D3c rsLoRA is worse on both splits => do NOT run round 2 from it. PiSSA-last DPO (dpo_D3c_pissa_a, train done in 60.8 min) evals running on GPU1.
