# configs/queues/rl/ (11 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/rl/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `rl4_g0.txt` | GPU0: reward configs c2 (no length) then c4 (KL beta 0.01); 150 steps each |
| `rl4_g1.txt` | GPU1: reward configs c1 (gốc) then c3 (cosine); 150 steps x 4 prompts x 8 gens each, vLLM, from merged v0; dev greedy + MBR8 after each |
| `rl5_g0.txt` | GPU0 (idle): c4f MBR8 (skipped earlier: duplicate session name), then c5 = cosine reward (best so far) with lr 2e-5; dev greedy + MBR8 |
| `rl6_g1.txt` | GPU1 (idle after ReAct): c6 = cosine reward + KL beta 0.01 (the two best ideas together); dev greedy + MBR8. Waits for other eval runs (tmux session names collide across GPUs). |
| `rl_c1.txt` | GRPO config c1 from v0 (merged), 50 steps x 4 prompts x 8 gens; then dev eval k5 (shots 0) — compare v0 .6026 on dev |
| `rl_c2.txt` | GRPO config c2 from v0 (merged), 50 steps x 4 prompts x 8 gens; then dev eval k5 (shots 0) — compare v0 .6026 on dev |
| `rl_smoke.txt` | GRPO smoke: 16 prompts, 10 steps, no vLLM, on GPU1 (GPU1 free). Measures time/memory and checks Qwen3.5 compatibility. |
| `rleval_g1.txt` | dev eval of GRPO c1 / c2 (v0 prompt, k5, shots 0); rescored on dev_clean / dev_strict afterwards |
| `rlfull_c1f.txt` | GRPO FULL (c1f): 300 steps x 4 prompts x 8 gens (~83% of train), vLLM colocate in .rl_env, from merged v0; then dev greedy + MBR8 on dev |
| `rlfull_c2f.txt` | GRPO FULL (c2f): 300 steps x 4 prompts x 8 gens (~83% of train), vLLM colocate in .rl_env, from merged v0; then dev greedy + MBR8 on dev |
| `rlsmoke_g1.txt` | GRPO with colocated vLLM, .rl_env (own venv); 10 steps; compare s/step (HF mode ~85-95 s per 32 completions) |
