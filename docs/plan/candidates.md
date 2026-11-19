# Candidate configurations (locked 2026-11-19; dev only, test not touched for these)

## C1 — v3b + 2 shots + MBR8  (configs/exp/A6_mbrv3b.yaml, run A6_mbrv3b__544aee20)
Pipeline: chunk `dieu` 2000/100 clean+dedup · AITeamVN embedding, dense top-50 · AITeamVN reranker, top-5 · Qwen3.5-2B SFT v3b (LoRA 3 epochs, 20 % no-answer rows) ·
prompt vi_rag + 2 nearest train shots · MBR over greedy + 7 samples (T 0.7), ROUGE-L consensus.
dev300 rougeL **.6575** (cos .911 jac .681 tok .756 bleu1 .609).

Paired bootstrap on dev (2,000 resamples, diff of per-question rougeL, 95 % CI):
| vs | dev300 | dev_clean | dev_strict |
|---|---|---|---|
| v0 greedy (.6026) | +.055 [+.032, +.079] | +.060 [+.033, +.087] | +.041 [+.001, +.080] |
| v0 + MBR8 (.6366) | +.021 [-.000, +.043] | +.019 [-.005, +.044] | -.001 [-.035, +.031] |

Reading: the gain over plain greedy is real; over v0 + MBR8 it is borderline (lower bound 0 on dev300, not significant on the clean/strict subsets) — MBR is doing most of the work.
Not yet measured: C1 on test (do ONCE at the end, "test-informed"); latency per stage (MBR8 = 8 samples per question).
Known better-in-parts, not yet combined: RL cosine (c3f greedy .6343 vs v0 .6026, +.032 CI .019..045) and v3b shots; GRPO from v3b is the next stacking experiment.

## C2 (provisional) — C1 with the weight-merged generator arith_v3b_c3f_a1  (run A6_mbrv3b__8f86afd5, 2026-11-20 05:28)
Same pipeline as C1, generator = task arithmetic v3b + 1.0·(c3f − v0) (src/train/merge.py arith, alpha 1; v3b SFT + the RL cosine delta), 2 shots, MBR8 (T 0.7).
dev300 rougeL **.6668** (C1 .6575). Paired bootstrap vs C1 MBR8: dev300 +.009 [-.004, +.022]; **dev_clean +.017 [+.001, +.035]; dev_strict +.021 [+.0005, +.043]** (both lower bounds > 0, barely).
Greedy (2 shots): .6516 vs C1 greedy .6325 (dev300 +.019 [+.006, +.034]; clean/strict CIs span 0).
Caveats: lower bounds are ~0, and ~10 variants were compared overnight (multiple comparisons) → treat as a lead, confirm with a second seed / the final train+dev retrain before replacing C1. Not run on test.
Other overnight null/negative results vs C1: oracle retrieval (+.002 ns), MBR16 (+.001 ns), MBR T0.5 (-.015), GRPO from v3b greedy .6271 (< C1 greedy), soup zero-shot .6301.

C2 confirmation (06:35): a05 merge MBR8 .6642 vs C1: dev_clean +.012 [+.001,+.025], dev_strict +.019 [+.004,+.038], dev300 +.007 ns. Seed repeat of C2 (same model, different sample) .6615: dev300 +.004 [-.010,+.017], dev_clean +.011 [-.009,+.029], strict +.001 [-.031,+.033] -> NOT significant on its own. Merged models are consistently ~+.004..+.009 above C1 under MBR but never clearly significant on dev300: C2 stays PROVISIONAL (C1 remains the safe choice). Third C2 pass and a second C1 pass (sampling noise reference) queued. MBR16 on arith_a1 died at 192/300 (queue kill) – not rerun.

Noise reference (06:57): a second MBR8 pass of C1 gives .6512 (vs .6575 first pass; MBR sampling noise ≈ ±.006). C2 passes: .6668 / .6615 / .6631 (mean .6638) vs C1 passes .6575 / .6512 (mean .6544): mean gap ≈ +.009, third C2 pass vs C1 first pass dev300 +.006 [-.009,+.021] (ns). Verdict: C2 is consistently a little above C1 but the effect is of the order of the sampling noise → C2 stays provisional; choose it for the final run only if the final train+dev retrain reproduces the gap.
