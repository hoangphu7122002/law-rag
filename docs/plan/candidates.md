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
