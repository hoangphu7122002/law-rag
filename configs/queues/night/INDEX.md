# configs/queues/night/ (59 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/night/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `night10_g1.txt` | c6f MBR8 was skipped (duplicate tmux session name with c5f's MBR8 run); rerun after the merged-model evals. |
| `night11_g1.txt` | ---- night2_g1 |
| `night12_g0.txt` | ---- night2_g0 (waits for the c3s2 queue 232412) |
| `night13_g1.txt` | (no description line) |
| `night14_g0.txt` | ---- night2_g0 (waits for the c3s2 queue 232412) |
| `night15_g1.txt` | (no description line) |
| `night16_g0.txt` | ---- night4_g0 |
| `night17_g1.txt` | GPU1 after react_s1 null result + oracle bound: kill T1 trajectories (low value); MBR8 of the merged model arith_v3b_c3f_a1 (2 shots) vs C1 MBR8 .6575, then oracle MBR8 with shots=2 (the first oracle MBR run had shots=0). |
| `night18_g0.txt` | GPU0: GRPO from v3b retry (OOM at step 7 with 5500 prompt tokens, 82 s/it): shorter prompts (3500), 100 steps; wait for it before serving. |
| `night19_g1.txt` | GPU1: does MBR16 add over MBR8 on the C1 model (v3b + 2 shots)? c3f: MBR8 .6450 -> MBR16 .6524. |
| `night1_g0.txt` | Night, GPU0 (after queue 221257 = c4f MBR8 + c5f): (1) teacher gate: raw Qwen3.5-9B on dev_clean, ReAct T0 and plain RAG (to beat: SFT v0 .583 plain, .571 ReAct); |
| `night20_g1.txt` | GPU1: MBR temperature mix (greedy + T0.4 x2 + T0.7 x2 + T1.0 x3) on v3b + 2 shots vs C1 MBR8 (greedy + 7x T0.7) .6575. |
| `night21_g1.txt` | GPU1: T1.0 samples hurt (.6197 vs .6575); test lower sampling temperature in MBR8: greedy + 7x T0.5, then 7x T0.3. |
| `night22_g0.txt` | GPU0: MBR sampling temperature with shots=2 EXPLICIT (the earlier T-mix/T0.5 runs forgot shots=2 -> shots 0, invalid). Uses A6_mbr (name differs from GPU1's A6_mbrv3b -> no tmux name collision). |
| `night23_g1.txt` | GPU1: MBR8 with shots=2 explicit for the merged arith_a1 (earlier run had shots 0) and for rl_v3b_c3 (its MBR eval was skipped by a name collision). |
| `night24_g1.txt` | GPU1: confirm C2 (arith_a1 + 2 shots + MBR8 = .6668): (1) a05 merge, (2) a seed-variance repeat of C2 (temps differ in the 8th sample so the run key is new). |
| `night25_g0.txt` | GPU0: C2 model with MBR16 (shots 2); A6_mbr16 name differs from GPU1's A6_mbrv3b. |
| `night26_g1.txt` | GPU1: third MBR sampling pass of C2 (arith_a1, shots2) to average out sampling noise. |
| `night27_g0.txt` | GPU0: second pass of C1 (v3b, shots2, MBR8) to measure the sampling noise of MBR itself (A6_mbr name; last temp .71 gives a new run key). |
| `night28_g1.txt` | GPU1: ReAct with the FORCE prompt (no 'only when insufficient' rule) and no initial passages (agent_initial_k=0), so the model must call tools. v0 first, then the raw 9B (is tool use there at all?), dev_clean. |
| `night29_g1.txt` | GPU1: same force prompt with kg_related added (3 of the 4 tools; web_search needs TINYFISH_API_KEY which is not on the box). |
| `night2_g0.txt` | Night, GPU0 (after the RL seed repeat 232412): trajectories seed 2. |
| `night2_g1.txt` | Night, GPU1 (after the compact-prompt eval 231703): collect ReAct trajectories of SFT v0 (T0 tools, T 0.7) on the TRAIN split, seed 1 (trajectories.jsonl in the run dir). Filter decontam at dataset-build time. |
| `night30_g1.txt` | GPU1: all 4 tools (search_law, get_article, kg_related, web_search) with the force prompt; web key read from a 600-mode file on the box (never in the repo). |
| `night31_g0.txt` | GPU0: ReAct trajectories on TRAIN with the force prompt (tool use required, no initial passages), tools search_law + get_article, seed 1, T0.7: raw material for S1 with real tool calls. Runs in parallel with GPU1's dev evals (per- |
| `night32_g0.txt` | GPU0 after the 2-tool trajectories: force-prompt trajectories on TRAIN with 3 tools (+kg_related, seed 2) then 4 tools (+web_search, seed 3; web is rate-limited to 500 calls/h by config). |
| `night33_g0.txt` | GPU0: does the RAW Qwen3.5-9B call tools when forced (force prompt, no initial passages)? SFT v0 never emits a tool call (0/166, rougeL .22 without context). |
| `night34_g1.txt` | GPU1: does the RAW (not SFT'd) Qwen3.5-2B call tools when forced? If yes, S1 can start from the raw 2B instead of v0. |
| `night35_g0.txt` | GPU0: raw 9B ReAct (force prompt, no initial passages) using the SHARED tools endpoint (:8300, one index copy). Waits for GPU1's 2B run (own index) to finish first so RAM is not doubled. |
| `night36_g1.txt` | GPU1: raw 2B ReAct (force prompt, min_tool_calls=1) through the shared tools endpoint started by the GPU0 queue. |
| `night37_g0.txt` | GPU0: raw Qwen3.5-9B as the teacher: ReAct trajectories on TRAIN (force prompt, no initial passages, T0.7, seed 1, tools search_law + get_article + kg_related) through the shared tools endpoint. min_tool_calls=1 so every trajector |
| `night38_g1.txt` | GPU1: raw Qwen3.5-9B as the teacher: ReAct trajectories on TRAIN (force prompt, no initial passages, T0.7, seed 1, tools search_law + get_article + kg_related) through the shared tools endpoint. min_tool_calls=1 so every trajector |
| `night39_g1.txt` | GPU1 after both teacher runs: build G1 data (gold final answers, 9B tool turns) and SFT the raw 2B; then eval on dev_clean + dev_strict (force prompt, no min_tool_calls) via the shared endpoint. |
| `night3_g1.txt` | Night, GPU1 after trajectories seed 1: T1 (T0 + kg_related, free) trajectories, seed 3, same settings. Keep separate from T0-only data (ablation). |
| `night40_g0.txt` | GPU0: raw Qwen3.5-9B as the teacher: ReAct trajectories on TRAIN (force prompt, no initial passages, T0.7, seed 1, tools search_law + get_article + kg_related) through the shared tools endpoint. min_tool_calls=1 so every trajector |
| `night41_g1.txt` | GPU1: react_g1 (tools used in 100% of questions, but its own search queries found the gold Dieu in only 66%: rougeL .651 when found, .315 when missed). Inference-side fix: every search_law call also retrieves with the original que |
| `night42_g1.txt` | GPU1: react_g1 + search_with_question with thinking ON (dev_clean only). Question: does a thought before the tool call raise the gold-Dieu hit rate (66-72%)? Compare with .5739 (thinking off). |
| `night43_g1.txt` | GPU1: generator ceiling. v3b + 2 shots, 8 candidates (greedy + 7 x T0.7) per dev question; greedy / MBR / best-of-n vs reference. No training. |
| `night44_g1.txt` | GPU1: DPO candidates. v3b samples (greedy + 7 x T0.7) on its own train prompts (refusal rows skipped); pairs built later by src/train/pref_data.py. |
| `night45_g1.txt` | GPU1: DPO smoke (16 pairs, 5 steps) on v3b. |
| `night46_g1.txt` | GPU1: DPO A (lr 2e-5, beta 0.1, 1 epoch, 1211 pairs from v3b own samples) -> eval greedy then MBR8 on dev300 (compare with C1: A5_v3b__177b2411 greedy, A6_mbrv3b__544aee20 MBR). |
| `night47_g1.txt` | GPU1: latency + FP8 for the DPO A model (alone on the GPU, concurrency 1 = per-question latency). dpo_a_bf16 / dpo_a_fp8 are symlinks of dpo_a (distinct run keys). Same weights; fp8 = vLLM --quantization fp8. |
| `night48_g1.txt` | GPU1: RAW legal models on dev (vi_rag, shots 0, greedy, cached retrieval). ViLegalQwen3 is gated (needs HF login) -> skipped. Control = raw Qwen3.5-2B. No external APIs. |
| `night49_g0.txt` | GPU0 (parallel to GPU1): apply the v3b recipe to Qwen2.5-3B-legal-vn (bqbbao6, 3.09B, raw dev .4553): SFT data with ITS template (no <think> block) -> SFT -> greedy (shots 2) -> MBR8. Compare with C1 (A5_v3b__177b2411 / A6_mbrv3b_ |
| `night4_g0.txt` | Night filler, GPU0 after trajectories seed 2: does MBR with 16 samples add over MBR8 on the best RL model (c3f, dev .6450 MBR8)? |
| `night50_g1.txt` | GPU1 (after the raw-model evals): DPO round 2 on v3b's own pairs (1211): lr 5e-5, beta 0.1, + 0.1 x SFT loss on the chosen answer (stabiliser); round 1 (lr 2e-5) gave greedy +.019 on dev300 but nothing on top of MBR. |
| `night51_g1.txt` | GPU1: iterative (on-policy) DPO. dpo_b (lr 5e-5 + SFT term) was WORSE than dpo_a (greedy dev300 .6421 vs .6543). Round 2 from dpo_a with its own fresh samples: sample 8 from dpo_a on the train prompts -> pairs -> DPO lr 2e-5 (as d |
| `night52_g0.txt` | GPU0: DPO round for the Qwen2.5-3B-legal-vn line (q25_v3b greedy .6153 is below C1 .6325; user asked to run DPO anyway). Samples from q25_v3b on its own train prompts -> pairs -> DPO lr 2e-5 (as dpo_a) -> greedy + MBR8. |
| `night53_g1.txt` | GPU1: iterative DPO round 3. dpo_a2 (round 2 from dpo_a, 898 pairs) gave greedy dev300 .6637 (+.031 [+.014,+.049] vs C1 greedy) and MBR8 .6721 (+.015 [-.001,+.030] vs C1 MBR). Round 3: sample from dpo_a2 -> pairs -> DPO lr 2e-5 -> |
| `night54_g1.txt` | GPU1: robustness of the dpo_a2 result. Round 3 (dpo_a3) DEGRADED (.6348 vs .6637, answers 2000 chars vs ref 1583) so the DPO rounds are noisy/overshoot; re-train round 2 (from dpo_a, same 898 pairs) with another seed -> dpo_a2s2 - |
| `night55_g0.txt` | GPU0: cheaper deployment mode for dpo_a2: MBR with 4 samples (greedy + 3 x T0.7) vs MBR8 (.6721) and greedy (.6637). |
| `night5_g1.txt` | Night filler, GPU1 after T1 trajectories (232839): ORACLE ceiling = gold Điều injected at rank 1 (retrieve.oracle=true), dev300, greedy and MBR8, for v3b + 2 shots (candidate C1 generator) and v0. |
| `night6_g0.txt` | Morning block, GPU0 after MBR16 (233314): STACKING = GRPO (cosine reward, refusal-aware) from SFT v3b (shots + 20 % no-answer rows), then dev greedy (2 shots) and MBR8 vs C1 (.6325 / .6575). |
| `night7_g1.txt` | GPU1 (replaces the T1 / oracle order): after trajectories seed 1 (232822): build S1 from them, LoRA-SFT from v0 (data/models/react_s1), then ReAct T0 on dev_clean with it (control: v0 ReAct .5709, plain RAG v0 .583). |
| `night8_g1.txt` | GPU1 after the ReAct SFT queue: T1 (+kg) trajectories seed 3, then ORACLE ceiling (gold Điều at rank 1) for v3b + 2 shots, greedy and MBR8. |
| `night9_g1.txt` | GPU1 after night8 (T1 trajectories + oracle): dev greedy of the merged models. soup (zero shot) vs c3f .6343 / v0 .6026; v3b+RL delta with 2 shots vs C1 greedy .6325. |
| `night9_prep.txt` | CPU only, now: build the merged models (soup of RL variants; v3b + (c3f - v0) task arithmetic, alpha 1 and .5). No GPU. |
| `night_g0.txt` | Night GPU0: SFT Qwen3.5-2B -> dev evals k5/k10 -> reranker data + fine-tune -> retrieval eval (dev) |
| `night_g1.txt` | Night GPU1: after train-vq2 -> clean copied base files -> evals -> Vi-Qwen2 raw k10 -> KD0 9B |
