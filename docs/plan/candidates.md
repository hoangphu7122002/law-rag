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
