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

## DPO round 1 on D3c PiSSA last-epoch (dpo_D3c_pissa_a) — 2026-11-26 01:05
rouge_l dev **.6405** vs base .6320 (+.0085); test **.6279** vs base .6252 (+.0027). Small gain, inside noise but positive on both splits (rsLoRA DPO was negative on both). Dev stages (n / rouge_l) DPO vs base: gold not in passages 25 / .370 vs .330; ref cites no article 36 / .505 vs .530; retrieved cites other Điều 24 / .461 vs 27 / .430; cites none 2 / .318 vs .323; retrieved and cited 213 / .718 vs .716. Refusal 1.3% (base 2.0%), len ratio 1.03. Pairs 1123 (margin .36). Reading: DPO helps the weak groups (a +.03, b +.04) without hurting the main group, for PiSSA but not for rsLoRA; effect size ~ +.003..+.009 overall, far below dpo_a2's +.03 on v3b. For comparison C PiSSA SFT alone: dev .6414 / test .6307.
## CPT v2 launched 2026-11-26 01:05
torchrun 2 GPU, mix.npy 209.2M tokens, 6320 steps at ~2.77 s/it => ~4h50, ETA ~05:55. SFT queues 1126-001308 (PiSSA data C, GPU0) and 1126-001311 (PiSSA data D3, GPU1) start automatically after cpt_report.json.

## CPT v2 finished — 2026-11-26 05:58
q35_cpt_v2: 1 epoch, 101,111 train sequences (209M tokens), 293 min on 2 GPU, train loss 1.228, eval loss 1.155 (v1 eval 1.293 was on a different mix, not comparable). SFT-from-CPT queues started 05:58: PiSSA on data C (GPU0, session train-q35cptC_pissa) and PiSSA on data D3 (GPU1, train-q35cptD3_pissa), 3 epochs, then dev/test of best-val and last. Reference: V_pissa (no CPT, data C) dev .6414 / test .6307; D3c PiSSA last (no CPT, D3) .6320 / .6252. Cloze probe of CPT v2 postponed until a GPU is free (probe needs ~6 GB; GPUs hold the SFT runs).

## SFT from CPT v2, PiSSA, data C (q35_cptC_pissa) — 2026-11-26 07:35
rouge_l (dev300 / test200): best-val **.6272 / .6112**; last-epoch **.6359 / .6298**. No-CPT reference V_pissa (same data/recipe): **.6414 / .6307**. => CPT effect: best-val -.014 / -.020, last -.005 / -.001 => decision rule (keep CPT only if dev >= +.01 and test not worse) FAILS. Training val loss was lower (best .309 vs .345) but that did not carry over to ROUGE-L.
Dev stages (n / rouge_l), V_pissa -> cptC best-val -> cptC last: gold not in passages 25 / .323 -> .279 -> .311; ref cites no article 36 / .574 -> .570 -> .585; retrieved cites other Điều 24 / .408 -> 26 / .462 -> 21 / .459 (+.05); retrieved cites none 4 / .299 -> 7 / .079 -> 7 / .256; retrieved and cited 211 / .724 -> .719 -> .713. Refusal rate 1.0% -> 5.3% -> 2.3% (CPT raised refusals, probably because the replay SFT rows / legal text taught "context lacks X"); len ratio 1.01.
Closed-book cloze of CPT v2 (209M tokens): train articles 51.0% (v1 54.1%, base 37.7%), dev/test gold raw-only 52.3% (v1 50.5%, base 34.5%): knowledge gain same as v1 despite 8x more tokens.
Reading: CPT injects facts (cloze +15 pts) and slightly helps the "cites another Điều" group (+.05) but adds refusals and loses elsewhere; net ROUGE-L neutral-to-negative. data-effect (C vs D3 from CPT) pending: cptD3_pissa training ends ~07:50, evals after.

## Diagnoses of 2026-11-26 (D3c rsLoRA dev300; scripts/analysis/diag_cited_group.py, scripts/analysis/diag_coverage.py)
- Cited-correct group (206): ROUGE-L by answer/reference length ratio: 0.7-1.3x n=116 F .841; >1.3x (over-quoting) n=64 F .546 (P .42 R .82); <0.7x n=26 F .561 (P .91 R .42). Reference names a khoản in 47/64 and 22/26 of the mismatches. Ceiling if mismatches reached .84: ~+.09 overall. V_pissa shows the same pattern (69 / 30 mismatches).
- Gold coverage by the context (article level): all gold retrieved 164 q .703; >=50% 46 q .635; <50% 29 q .557; none 25 q .331 (41% of lost points are in the 100 not-fully-covered questions; ceiling +.055). Mean gold-article recall .744.
- Long gold articles: ROUGE-L by longest gold article: <1000 chars n=68 .737; 1000-2000 n=59 .722; 2000-4000 n=52 .627; >4000 n=83 .515. 44% of dev gold articles are >2000 chars (all articles 20%). Reference 4-gram coverage by context vs full gold: <=2000 chars no loss; 2000-4000 .677 vs .756; >4000 .602 vs .699 (only 33% of a >4000-char article's text is in the context).
## Full-Điều context test (no chunk cutting), dev300 rouge_l, 2026-11-26 08:25
D3c rsLoRA: chunks (k=5) .6258 | 203 (whole articles of the top-5 chunks) .6123 | 201 (5 distinct articles of top-15 chunks, whole) .6173 | 202 (8 articles) .6097. V_pissa: chunks .6414 | 201 .6306. => whole-article context is WORSE (-.009..-.016) for both models, so "do not cut long articles" does not help in this pipeline (more text per article -> more over-quoting / noise). Not pursued; sentence-boundary cut not tested.
## SFT from CPT, data D3 (cptD3_pissa), 2026-11-26 08:25
best-val dev .5987 / test .5603; last dev .6200 / test .5824 — clearly worse than no-CPT D3c PiSSA (.6225-.6320 / .5987-.6252). CPT (209M tokens) hurts ROUGE-L here; CPT closed as a negative result.
## dpo_a2 re-eval under today's pipeline: dev .6637 / test .6433 (identical to CHANGELOG) => best model so far; pipeline comparable.

## DPO on V_pissa and extent-aware control, 2026-11-26 09:45 (rouge_l)
- dpo_vb_ext (v3b + extent-aware pairs, round 1; same cands as dpo_a): dev .6526 / test .6298. dpo_a (standard pairs, round 1): dev .6514/.6543 (CHANGELOG). => no gain from the khoản/length reward in round 1 on v3b.
- dpo_Vp_A (V_pissa + standard pairs, 1175 pairs, round 1): dev **.6493** (V_pissa .6414, +.008) / test **.6226** (V_pissa .6307, -.008). Comparable with dpo_a round 1 (.6514-.6543 dev); dpo_a2 needed round 2 to reach .6637/.6433. Round 2 from dpo_Vp_A queued on GPU1 (queue dpoVp2_g1: sample n=8, pairs, DPO, dev/test); dpo_Vp_B (extent-aware pairs) training on GPU0.

## 10:45 — dpo_Vp_B and paired bootstrap (scripts/analysis/paired_bootstrap.py, dev+test pooled n=500, rouge_l)
dpo_Vp_B (V_pissa + extent-aware pairs): dev .6450 / test .6154 (dpo_Vp_A std pairs .6493 / .6226) => the khoản/length reward does NOT help (also null on v3b: dpo_vb_ext .6526/.6298 vs dpo_a .6514-.6543).
Pooled dev+test mean rouge_l and paired-bootstrap difference vs dpo_a2 (mean .6556): V_pissa .6371 (-.0185 [-.0327,-.0042]); dpo_Vp_A .6386 (-.0170 [-.0306,-.0035]); dpo_Vp_B .6331 (-.0224 [-.0365,-.0081]); dpo_vb_ext .6435 (-.0121 [-.0240,-.0007]); D3c PiSSA last + DPO .6354 (-.0202 [-.0347,-.0060]); D3c PiSSA last .6293 (-.0263 [-.0417,-.0111]); D3c rsLoRA .6248 (-.0308 [-.0457,-.0157]). => dpo_a2 is significantly better than every other candidate (95% CIs exclude 0, dpo_vb_ext borderline). Round 2 from dpo_Vp_A (dpo_Vp_A2) training, ETA eval ~11:10. Wide-context (k=102..105) eval of dpo_a2 queued on GPU0 (queue 1126-104116).

## 11:10 — NEW BEST on dev: dpo_Vp_A2 (V_pissa -> DPO round 1 -> DPO round 2, standard pairs), rouge_l
dev **.6743** / test **.6435** vs dpo_a2 dev .6637 / test .6433 (dev +.0106, test +.0002). Pooled dev+test (n=500) .6620 vs dpo_a2 .6556: +.0064 [-.0066,+.0197] (not significant); vs dpo_Vp_A (round 1) +.0234 [+.0112,+.0355] (round 2 is what matters, as with dpo_a -> dpo_a2). Pairs of round 2: 1043 (margin .284, chosen is greedy 12%).
Wide-context eval of dpo_a2 (dev, retrieve.k markers): base k=5 .6637 | 102 (gate->top10) .6609 | 103 (gate->top15) .6629 | 104 (always top15) .6545 | 105 (oracle widen) .6667 => widening context does not help dpo_a2 (oracle upper bound only +.003).
Seed replicates queued: round 2 with DPO seed 777 (GPU1, dpo_Vp_A2s2) and a full second chain (round 1 + sampling + round 2, seed 777, GPU0, dpo_Vp_As2 / dpo_Vp_As2_r2).

## 12:05 — seed replicate of round 2, soup (rouge_l)
- **dpo_Vp_A2s2** (round 2, DPO seed 777, same pairs/base dpo_Vp_A): dev **.6782** / test **.6472** (dpo_Vp_A2: .6743 / .6435; dpo_a2 .6637 / .6433). The round-2 step reproduces (dev spread .004, test .004). Pooled dev+test .6658 vs dpo_a2 .6556: +.0102 [-.0020,+.0231] (p~.1); dpo_Vp_A2 +.0064. Tail/mid dev (n / rouge_l): cited-correct 214 / .754, cites other 21 / .475, gold missing 25 / .371, ref no article 36 / .591, refusal .3%, len ratio .97.
- dpo_Vp_As2 (round 1, seed 777; the replicate chain on GPU0): dev .6544 (dpo_Vp_A .6493). Test pending; round 2 of that chain follows.
- **soup_a2 = mean(dpo_a2, dpo_a2s2)**: dev .6632 / test .6346 (dpo_a2 .6637/.6433, dpo_a2s2 .6623/.6377); pooled .6518 vs dpo_a2 -.0038 [-.0111,+.0032] => weight averaging of the DPO seeds does NOT help (dropped).
- MBR8 of dpo_Vp_A2 (A6_mbr, shots 2) running on GPU1.

## 12:31 — MBR8 of dpo_Vp_A2, round 1 seed replicate
- **dpo_Vp_A2 + MBR8** (A6_mbr: greedy + 7 x T0.7, ROUGE-L consensus, shots 2): dev **.6822** (greedy .6743, +.008) / test **.6446** (greedy .6435, +.001). dpo_a2 + MBR8 had test .6521 (+.009 over greedy .6433). => MBR adds ~+.001..+.009; on the best model it does not move test. Not worth 8x inference cost unless accuracy mode is wanted.
- dpo_Vp_As2 (round 1, seed 777): dev .6544 / test .6263 (dpo_Vp_A .6493 / .6226): round 1 also reproduces (spread +.005 / +.004).
- GPU0: dpo_Vp_As2_r2 (round 2 of the second chain) training; GPU1: round 3 sampling from dpo_Vp_A2 (queue 1126-122633).
- Disk 67G after deleting the CPT models and soup_a2 with the user's approval (31G freed).

## 2026-11-26 13:00 — pair coverage by bucket (pref_Vp2 cands, pairs_mix 1323)
Bucket = mean ROUGE-L of the 8 samples (tail <.45, mid .45–.65, head ≥.65).
| bucket | prompts | pairs | pairs/prompt | no pair | extent/synth pairs |
|---|---|---|---|---|---|
| tail | 346 | 292 | 0.84 | 73 | 101 |
| mid | 299 | 402 | 1.34 | 56 | 239 |
| head | 807 | 629 | 0.78 | 310 | 283 |
Read: tail under-covered (0.84/prompt, 21% no pair); head wastes 283 extent pairs. Gap for tail+mid ≈ 130 prompts with no pair (+ ~150 more tail pairs for parity). Haiku 5.5 low candidate targets: those prompts (chosen = Haiku answer verified by rougeL vs gold ≥ .65, rejected = model's worst); decontaminate vs dev/test (train split is disjoint by qid; also check question text). Queued dpo_Vp_A2m (queue 1126-125335) on GPU0 after chain-2.

## 2026-11-26 13:40 — full chain-2 replicate and round 3 (rouge_l; models verified via config.yaml)
| model | dev | test | pooled n=500 | diff vs dpo_a2 (95% CI) | dev len ratio |
|---|---|---|---|---|---|
| dpo_Vp_A2 (chain 1) | .6743 | .6435 | .6620 | +.0064 [-.0066,+.0197] | .96 |
| dpo_Vp_As2_r2 (chain 2, all seeds 777, from scratch) | .6586 | .6299 | .6471 | -.0085 [-.0216,+.0049] | .99 |
| dpo_Vp_A3 (round 3 from A2) | .6453 | .6166 | .6338 | -.0217 [-.0351,-.0085] | 1.11 |
Read: the two-round chain does NOT reproduce from scratch (chain 2 is -.016 dev vs chain 1; round-1 did reproduce .6544/.6263). Round-2 gain depends on the round-1 sample set (seed noise ~±.01-.015). Round 3 is significantly worse and over-long (len ratio 1.11, ref-cites-no-article .554, other-Điều .445) → stop at 2 rounds. dpo_Vp_A2m (mixed extent pairs) is running on GPU0.
