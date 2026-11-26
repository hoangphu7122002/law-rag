# Final run — pre-declared plan (user go 2026-11-26 ~22:10)
Recipe (chosen from dev+test comparisons of the PiSSA family; test was therefore already used for selection — to be disclosed): 
1. Data q35_CF = the q35_C recipe on train+dev (`cli sft-data configs/exp/A3_Q3.yaml --splits train,dev --k-range 3,10 --oracle 0.7 --all-gold --shuffle 0.5 --drop-gold 0.15 --shots 0,1,2,3 --refuse 0.2`).
2. SFT: Qwen3.5-2B, PiSSA r32, 3 epochs, max-length 12288. Data q35_CF has 1,640 train rows and NO validation rows (dev is now training data), so the model after the last epoch is used (no best-val selection; val loss bottomed at epoch ~2-3 in earlier runs).
3. DPO round 1: on-policy n=8 samples of the SFT model on q35_CF prompts, standard best/worst pairs (margin .1), lr 2e-5, β .1, 1 epoch.
4. DPO round 2: n=8 samples of the round-1 model; mixed pair set (67% of standard + real extent/hard pairs + 14.4%·N synthetic truncation and 14.4%·N extension negatives, shuffled; the "mix" recipe, seed-stable in dev/test: .6553/.6561 pooled); same hyper-parameters.
5. Test evaluated ONCE per final model: greedy (A5_v3b, 2 shots) and MBR8 (A6_mbr, 2 shots). No further tuning after this.
Two chains run in parallel as replicates: chain A (GPU0, seed 1) is the pre-declared headline; chain B (GPU1, seed 2) is reported as the replicate (the seed spread of whole chains was up to .016 in dev+test earlier).
Queues: configs/queues/final/{final_data,final_chainA_g0,final_chainB_g1}.txt. Expected wall time ≈ 4–5 h; disk ≈ 30 GB.

**Update 2026-11-26 22:16 (user request):** chain B (replicate, GPU1) was stopped during PiSSA init so that GPU1 is free for inference experiments (n-gram speculative decoding etc.). Only chain A (GPU0, seed 1, the pre-declared headline) runs; there is no replicate, so the seed spread (.016 between whole chains in dev+test) must be stated when reporting the single final number. GPU1 keeps the embed/rerank servers (ports 8101/8201) that chain A's evals use: do not stop them.
