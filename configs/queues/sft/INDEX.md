# configs/queues/sft/ (35 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/sft/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `D2_build.txt` | Data D2: like D but k random 5-15 and EVERY gold article in the context IN FULL (--full-gold: chunks of a gold article replaced by the whole article, all missing gold articles added, context grows when needed); no drop-gold; 764 s |
| `D2c_g0.txt` | CONTINUE training the finished C model (q35_V_pissa, learned from at most 10 passages) on data D2 (k 5-15, every gold article in full at random positions): 2 epochs, lr 5e-5, seed 1, --save-last --keep-ckpt; dev+TEST ROUGE-L of th |
| `D2c_g1.txt` | CONTINUE training the finished C model (q35_V_rslora, learned from at most 10 passages) on data D2 (k 5-15, every gold article in full at random positions): 2 epochs, lr 5e-5, seed 1, --save-last --keep-ckpt; dev+TEST ROUGE-L of t |
| `D3_build.txt` | Data D3 = D2 without the refusal-heavy parts: D/D2 models over-refused (dev 2-3% refusals vs 0.7-1% for C; each refusal scores ~0.07 vs ~0.64) and scored .02-.03 below C. Synthetic no_answer rows removed (data/synth_v1_answerable. |
| `D3c_g0.txt` | CONTINUE training the finished C model (q35_V_pissa) on data D3 (k 5-15, every gold article in full at random positions, 5% refusal rows): 2 epochs, lr 5e-5, seed 1, --save-last --keep-ckpt; dev+TEST ROUGE-L of the best-val and th |
| `D3c_g1.txt` | CONTINUE training the finished C model (q35_V_rslora) on data D3 (k 5-15, every gold article in full at random positions, 5% refusal rows): 2 epochs, lr 5e-5, seed 1, --save-last --keep-ckpt; dev+TEST ROUGE-L of the best-val and t |
| `D3c_g1_last.txt` | D3c rsLoRA last-epoch evals (dev then test); best-val evals already done. |
| `D3c_g1_last_test.txt` | D3c rsLoRA last-epoch TEST eval (dev done .6596); waits for GPU1 memory to be released. |
| `D3c_g1_resume.txt` | D3c rsLoRA remaining evals (queue 180423 stopped at the 2nd serve: GPU1 memory not yet released after the first eval). Test best-val, then dev+test last-epoch. |
| `D_build.txt` | Full build of data D (guarantee, k 3-15, shuffle 0.7, 764 synthetic rows). Launch only after the smoke test is clean. |
| `D_g0.txt` | Data D, pissa, seed 1, 3 epochs, --save-last: dev300 + TEST ROUGE-L of the best-val checkpoint and of the last-epoch model (user-requested; test numbers are used to choose the training method). GPU0 lane. Waits for data D (D_build |
| `D_g1.txt` | Data D, rslora, seed 1, 3 epochs, --save-last: dev300 + TEST ROUGE-L of the best-val checkpoint and of the last-epoch model (user-requested; test numbers are used to choose the training method). GPU1 lane. Waits for data D (D_buil |
| `D_smoke.txt` | Smoke test of data D: first 30 train questions + 30 synthetic rows, same flags as D_g0. Check report.json before the full build. |
| `a4_g0.txt` | A4 SFT v0, GPU0: train retrieval -> SFT data (both models) -> smoke -> train Qwen3.5-2B -> eval dev |
| `a4_g1.txt` | A4 SFT v0, GPU1: after the A3 round-2 queue and the vq2 data -> train Vi-Qwen2 -> eval dev |
| `bc70_g0.txt` | SFT B and C (2 seeds each; this lane = seed 1, bc70_g1 = seed 2). B = v3b recipe + shuffle 50 % + inject ALL missing gold; C = B + 15 % rows with gold dropped (retrieval miss, reference kept). Dev300 greedy, 2 shots. Models kept u |
| `bc70_g1.txt` | SFT B and C, seed 2 (GPU1 lane; embed :8101 + rerank :8201 stay up). Waits for the data built by bc70_g0. |
| `ep6_g0.txt` | 6-epoch SFT (best val-loss checkpoint kept): tests whether 3 epochs cut the runs short (B/C best val was at the last step). GPU0 lane: A6_s1. |
| `ep6_g1.txt` | 6-epoch SFT (best val-loss checkpoint kept): tests whether 3 epochs cut the runs short (B/C best val was at the last step). GPU1 lane: C6_s1. |
| `hp_g0.txt` | SFT hyper-parameter A/B on the v0 data (datamate H200 branch): r32 alpha64 dropout0 lr2e-4, 2 epochs, val every 30 steps |
| `kd0.txt` | KD0: Qwen3.5-9B teacher on dev, GPU1 after the A4 GPU1 queue |
| `lc61_g0.txt` | Learning curve (Track A): the v3b SFT recipe on 25% and 50% of the train questions, two seeds each (subset seed = training seed); 100% = q35_v3b (seed 1, .6325) and q35_v3b_s2 (.603, deleted). Dev300 greedy, 2 shots. Models are de |
| `n2_g0.txt` | GPU0 (rebuilt): v3b train already running (lawrag-run-train-q35v3b) -> eval v3b -> eval hp A/B (q35_h) |
| `n2_g1.txt` | GPU1 (rebuilt 07:58: old `wait` used a truncated tmux session name -> server killed under the eval; now distinct exp stems A5_*) |
| `n3_g0.txt` | GPU0 (rerun after "GPU 0 free < 20000" race right after the kill): hp A/B eval, then v3b without shots / with 3 shots |
| `seed57_g0.txt` | SFT seed check: the v3b recipe re-trained with seed 777 (everything else identical) -> greedy and MBR8 on dev300. Tells how much of the v3b numbers is training noise. |
| `seed64_g0.txt` | Seed variance of the full-data v3b recipe: a third SFT seed (555) on 100% of train, dev300 greedy 2 shots (seeds 1: .6325, 777: .603). Model deleted after its eval. |
| `v1_g1.txt` | GPU1: SFT v1 = Qwen3.5-2B, prompt cancu, 3 epochs (best val ~ step 270 in v0) -> dev eval k5/k10 |
| `v2_g1.txt` | SFT v2 (user idea): varied k in 5..10 + gold passage injected (p .7) when top-k misses it -> eval dev k5 / k10 |
| `v3_g1.txt` | SFT v3 (user request): few-shot rows (0-3 shots, nearest decontaminated train Q/A, never the row's own Dieu), dynamic k 3..10, |
| `v3b_g0.txt` | SFT v3b: same as v3 but 20 % no-answer rows (user: more no-answer cases) -> compare refusal rate vs rougeL on dev |
| `var2_g0.txt` | SFT variant study on the C data, seed 1, 3 epochs, GPU0 lane (DoRA dropped: NaN loss from step 1). Compare val loss + dev300 ROUGE-L with C seed 1 (best val .3640, rougeL .6202). Models kept until compared. |
| `var3_g0.txt` | lr 2e-5 on the C data (LoRA r32, seed 1, 3 epochs), GPU0 lane. Compare with C seed 1 (lr 1e-4: val .3640, rougeL .6202). |
| `var_g0.txt` | SFT variant study on the C data, seed 1, 3 epochs, GPU0 lane. Compare val loss + dev300 ROUGE-L with C seed 1 (best val .3640, rougeL .6202). Models kept until compared. |
| `var_g1.txt` | SFT variant study on the C data, seed 1, 3 epochs, GPU1 lane. Compare val loss + dev300 ROUGE-L with C seed 1 (best val .3640, rougeL .6202). Models kept until compared. |
