# configs/queues/dpo/ (11 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/dpo/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `abl58_g0.txt` | GPU0 (after the SFT seed check): DPO beta ablation, beta .05 and .3 (round 1 from v3b, margin .1, lr 2e-5; reference dpo_a = beta .1 .6543). |
| `abl58_g1.txt` | GPU1 ablations for the report: (1) k and shots of the FINAL model (dpo_a2), (2) DPO design: pair margin threshold .05 / .2 (round 1 from v3b, lr 2e-5, beta .1; reference dpo_a = margin .1 .6543). |
| `cw63_g0.txt` | Track C: stronger cited-Dieu term in the DPO reward (round 1 only, from SFT v3b, lr 2e-5, beta .1, margin .1): cite_weight 0.4 and 0.6 vs the original 0.2 (dpo_a, dev300 .6543). Same candidates (data/train/pref_v3b/cands.jsonl). |
| `dpoVp2_g1.txt` | DPO round 2 from dpo_Vp_A (round 1: dev .6493 / test .6226; V_pissa .6414 / .6307). The dpo_a2 recipe needed two rounds (dpo_a round 1 dev .6514/.6543 -> dpo_a2 .6637): on-policy samples of round 1, standard pairs, lr 2e-5, beta . |
| `dpoVp_g0.txt` | DPO B on V_pissa (extent-aware pairs) on GPU0: waits for the full-article eval queue and for pairs_B.jsonl (made by queue dpoVp_g1), then dev + test. |
| `dpoVp_g1.txt` | DPO on V_pissa (best SFT, C PiSSA dev .6414 / test .6307), dpo_a recipe. GPU1: sample n=8 on the C prompts, pairs A (standard reward) and B (extent-aware: +.15 khoản match, +.15 length 0.7-1.3x), DPO A here; DPO B runs on GPU0 (qu |
| `dpo_D3cP_r1_g1.txt` | DPO round 1 from D3c PiSSA last-epoch (dev .6650 / test .6680), GPU1, same recipe as dpo_D3c_a (rsLoRA, GPU0). |
| `dpo_D3c_r1_g0.txt` | DPO round 1 from the D3c rsLoRA best-val model (dev .6657 / test .6642): on-policy samples n=8 on the D3 train prompts, pairs margin>=.1, DPO lr 2e-5 beta .1 (the dpo_a recipe), then dev300 + test200. |
| `dpo_a2_reeval_g0.txt` | Re-evaluate dpo_a2 (v3b + 2 DPO rounds, the old best) under today's pipeline for an apples-to-apples DPO comparison (dev300, then test200). |
| `dpo_vb_ext_g0.txt` | Controlled test of the extent-aware reward on the SAME samples that made dpo_a (v3b cands, round 1; dpo_a dev .6514/.6543 with standard pairs): pairs with +.15 khoản match, +.15 length band -> DPO from q35_v3b (lr 2e-5, beta .1, d |
| `repl65_g0.txt` | Replication of the headline result from another SFT seed: iterative on-policy DPO (2 rounds, same recipe as dpo_a / dpo_a2) starting from lc_f100_s555 (SFT seed 555, dev300 greedy .6134). Compare with the original chain: SFT .6325 |
