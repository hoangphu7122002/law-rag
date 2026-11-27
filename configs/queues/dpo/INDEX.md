# configs/queues/dpo/ (23 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/dpo/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `abl58_g0.txt` | GPU0 (after the SFT seed check): DPO beta ablation, beta .05 and .3 (round 1 from v3b, margin .1, lr 2e-5; reference dpo_a = beta .1 .6543). |
| `abl58_g1.txt` | GPU1 ablations for the report: (1) k and shots of the FINAL model (dpo_a2), (2) DPO design: pair margin threshold .05 / .2 (round 1 from v3b, lr 2e-5, beta .1; reference dpo_a = margin .1 .6543). |
| `anchor_g1.txt` | Anchored DPO round 1 (user 2026-11-27): 19 % of the round-1 pairs teach a chosen answer below ROUGE-L .5. Same cands as dpo_a (no resampling), from q35_v3b, dev greedy. |
| `cw63_g0.txt` | Track C: stronger cited-Dieu term in the DPO reward (round 1 only, from SFT v3b, lr 2e-5, beta .1, margin .1): cite_weight 0.4 and 0.6 vs the original 0.2 (dpo_a, dev300 .6543). Same candidates (data/train/pref_v3b/cands.jsonl). |
| `dpoVp2_g1.txt` | DPO round 2 from dpo_Vp_A (round 1: dev .6493 / test .6226; V_pissa .6414 / .6307). The dpo_a2 recipe needed two rounds (dpo_a round 1 dev .6514/.6543 -> dpo_a2 .6637): on-policy samples of round 1, standard pairs, lr 2e-5, beta .1. GPU1. |
| `dpoVp_A2c_g1.txt` | Cite-contrast DPO: round 2 from dpo_Vp_A on the standard pairs (1043) + 143 cite-contrast pairs (best sample citing a gold Điều vs worst sample citing only other Điều, ROUGE margin >= .05), shuffled (pairs_contrast.jsonl). Targets the 'right law, wrong article' failure. Compare with dpo_Vp_A2 (.6743/.6435), A2ms2 (.6663/.6410), dpo_a2. GPU1. |
| `dpoVp_A2cs2_g1.txt` | Seed-777 replicate of the cite-contrast DPO (dpo_Vp_A2c: dev .6728 / test .6413; pooled .6602). GPU1. |
| `dpoVp_A2g_g1.txt` | Round 2 from dpo_Vp_A with pairs_mix_gold (mixed extent pairs + 129 gold-as-chosen tail/mid pairs, decontaminated vs dev/test, shuffled). GPU1 after the A2ms2 queue. |
| `dpoVp_A2m_eval_g0.txt` | resume: dev/test eval of dpo_Vp_A2m (training done 13:55; the wait pattern matched dpo-VpA2ms2, so the first queue stalled) |
| `dpoVp_A2m_g0.txt` | Round 2 from dpo_Vp_A with the mixed pair set (700 of 1043 standard + 323 real extent/hard pairs + 300 synthetic truncation/extension negatives, shuffled; data/train/pref_Vp2/pairs_mix.jsonl). Compare with dpo_Vp_A2 (.6743/.6435) and A2s2 (.6782/.6472). GPU0, after the seed chain. |
| `dpoVp_A2ms2_g1.txt` | Seed-777 replicate of dpo_Vp_A2m (mixed extent pairs, round 2 from dpo_Vp_A). GPU1. |
| `dpoVp_A2n16_g0.txt` | Round 2 from dpo_Vp_A with n=16 on-policy samples per prompt (instead of 8): more candidates -> better best/worst pairs (tail-mid plan, user asked about n=16). Plain pair recipe otherwise. Compare with dpo_Vp_A2 (.6743/.6435; chain spread ~.016), A2ms2, dpo_a2. GPU0. |
| `dpoVp_g0.txt` | DPO B on V_pissa (extent-aware pairs) on GPU0: waits for the full-article eval queue and for pairs_B.jsonl (made by queue dpoVp_g1), then dev + test. |
| `dpoVp_g1.txt` | DPO on V_pissa (best SFT, C PiSSA dev .6414 / test .6307), dpo_a recipe. GPU1: sample n=8 on the C prompts, pairs A (standard reward) and B (extent-aware: +.15 khoản match, +.15 length 0.7-1.3x), DPO A here; DPO B runs on GPU0 (queue dpoVp_g0). Then dev + test. |
| `dpoVp_r3_g1.txt` | Third DPO round from dpo_Vp_A2 (dev .6743 / test .6435; replicate A2s2 .6782 / .6472): on-policy n=8 on the C prompts, standard pairs, same recipe. On v3b round 3 degraded (dpo_a3 dev .6348, answers longer): watch len_ratio. Disk freed to 67G (CPT models deleted with the user's approval). GPU1 after the MBR queue. |
| `dpoVp_seed_r2_g1.txt` | Seed check of the round-2 step that made dpo_Vp_A2 (dev .6743 / test .6435): same pairs (pref_Vp2/pairs.jsonl, from dpo_Vp_A), DPO seed 777 -> dpo_Vp_A2s2, dev + test. GPU1. |
| `dpoVp_seedchain_g0.txt` | Full replicate of the V_pissa DPO recipe with another seed: round 1 (pairs_A of the first chain, DPO seed 777) -> dev/test -> on-policy sampling n=8 from it on the C prompts -> pairs -> round 2 (seed 777) -> dev/test. Compare with dpo_Vp_A (.6493/.6226) and dpo_Vp_A2 (.6743/.6435). GPU0. |
| `dpo_D3cP_r1_g1.txt` | DPO round 1 from D3c PiSSA last-epoch (dev .6650 / test .6680), GPU1, same recipe as dpo_D3c_a (rsLoRA, GPU0). |
| `dpo_D3c_r1_g0.txt` | DPO round 1 from the D3c rsLoRA best-val model (dev .6657 / test .6642): on-policy samples n=8 on the D3 train prompts, pairs margin>=.1, DPO lr 2e-5 beta .1 (the dpo_a recipe), then dev300 + test200. |
| `dpo_a2_reeval_g0.txt` | Re-evaluate dpo_a2 (v3b + 2 DPO rounds, the old best) under today's pipeline for an apples-to-apples DPO comparison (dev300, then test200). |
| `dpo_vb_ext_g0.txt` | Controlled test of the extent-aware reward on the SAME samples that made dpo_a (v3b cands, round 1; dpo_a dev .6514/.6543 with standard pairs): pairs with +.15 khoản match, +.15 length band -> DPO from q35_v3b (lr 2e-5, beta .1, default max-length) -> dev + test. GPU0. |
| `repl65_g0.txt` | Replication of the headline result from another SFT seed: iterative on-policy DPO (2 rounds, same recipe as dpo_a / dpo_a2) starting from lc_f100_s555 (SFT seed 555, dev300 greedy .6134). Compare with the original chain: SFT .6325 -> dpo_a .6543 -> dpo_a2 .6637. |
| `rft_VpA_g0.txt` | Best-of-8 rejection-sampling SFT (expert iteration) from dpo_Vp_A on its own round-2 samples (data/train/rft_Vp2: best-ROUGE sample per prompt, length .8-1.3x, rouge>=.55). LoRA r32, lr 2e-5, 1 epoch. Compare with dpo_Vp_A2 (DPO round 2: .6743/.6435) and dpo_a2. GPU0 after the oracle eval. |
