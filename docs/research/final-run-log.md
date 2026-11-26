# Final run log
- 2026-11-26 21:58: q35_CF built: 2545 → 1640 train rows after decontamination (train 1452 + 188 from dev), 0 val rows. SFT of both chains started (no validation → last-epoch model; recorded in final-run-plan.md).
- 2026-11-26 22:14: both SFT jobs are in PiSSA initialisation (full SVD on CPU, ~31 cores each, load average 126, GPUs idle) after 15 min; the V_pissa runs used the same full-SVD init. Waiting; if still no training step by ~22:50 consider restarting chain B after chain A's init to remove CPU contention (not changing the recipe).
- 2026-11-26 22:16: chain B stopped (user wants GPU1 for inference tests); chain A (sftFA, GPU0) continues, CPU contention gone. GPU1 embed/rerank stay up (chain A eval needs ports 8101/8201).
