# Embedder fine-tuning (AITeamVN/Vietnamese_Embedding, Modal, 2026-11-19)

Data: data/train/rr_v0/train.jsonl = 4,015 groups (course train + Zalo, decontaminated vs test), 1 positive + 7 hard negatives. Loss: MNRL (CachedMNRL, batch 128).
Eval: EmbedBench on the dev pool (264 questions; dev_clean 130, dev_strict 61), paired bootstrap vs the untouched base (rule stated BEFORE the second grid:
lower 95 % CI > 0 on all / dev_clean / dev_strict for Điều@1, no drop at Điều@20, seed repeat agrees).

| variant | Điều@1 | Điều@20 | diff @1 all [95 % CI] | clean | strict |
|---|---|---|---|---|---|
| base | .534 | .955 | | | |
| lora32_t02_low (lr 2e-5, scale 50) | .557 | .955 | +.023 [+.004, +.045] | +.031 [+.008, +.062] | +.016 [.000, +.049] |
| lora8_t02 (r 8, scale 50) | .557 | .955 | +.023 [+.004, +.045] | +.023 [.000, +.054] | +.016 [.000, +.049] |
| full_low@0.5 (interpolated) | .549 | .955 | +.015 [-.004, +.034] | +.015 [.000, +.038] | +.016 [.000, +.049] |
| lora32_low@0.5 | .545 | .955 | +.011 [-.004, +.030] | +.015 [.000, +.038] | +.016 [.000, +.049] |
| lora32_t02 (lr 1e-4) | .557 | .947 | +.023 [-.011, +.057] | +.031 [-.023, +.085] | .000 [-.082, +.082] |
| lora32_t02_s2 (seed repeat) | .542 | .955 | +.008 [-.030, +.042] | +.015 [-.038, +.062] | -.016 [-.082, +.049] |
| query32 (query tower only) | .542 | .943 | +.008 [-.034, +.049] | +.015 [-.046, +.077] | -.016 [-.115, +.082] |
| lora32_low, lora32_vlow, full_low | .534-.538 | .943-.955 | ~0 | ~0 | ~0 |
| first grid (lr 1e-4, scale 20; no per-question ranks): lora32 .485, lora64 .462, full .481, e3 .485, inbatch .519, lora8 .534 | | | worse than base | | |

## Verdict
- Fine-tuning with lr 1e-4 / scale 20 DAMAGES the base (AITeamVN saw ~1M triplets; our 4k groups pull it off). Conservative settings (lr ≤ 2e-5, scale 50) are neutral to slightly positive.
- Best cells (+.023 Điều@1, +.014 MRR) pass on all 264 and on dev_clean but touch 0 on dev_strict (lower bound .000, not > 0) and the seed repeat of the lr 1e-4 cell did not reproduce → by the pre-stated rule NOT accepted.
- Even if accepted: +.02 at Điều@1 on a pool built from AITeamVN's own top-50; recall is not the bottleneck (Điều@20 .955) and earlier retrieval gains (rr_v0 Điều@1 +.125) did not move generation rougeL. No re-index of 724k.
- Decision: stop embedder training. Query-only tuning (index stays valid) was no better. Effort goes to the generator side (RL, MBR, ReAct data).

Cost: ~15 scored variants x (train 1-13 min + ~3 min load/eval) ≈ 235 GPU-min on L40S ≈ $7.6 (+ ~$0.3 lost to one stopped duplicate launch), Modal profile phuchuynhvinh-fwork.
