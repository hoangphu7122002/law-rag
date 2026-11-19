# Findings: algorithm config (agent summary from snippets, unverified; check odd arXiv ids)
- Seen on 1.5B models: lr 1e-6 (full FT), G 4–8, beta 0–0.01 (0.04 hurt), ~2000 steps, LoRA rank 8–32 (arxiv 2605.12207, 2512.07611, 2606.30789). Dynamic sampling only inside DAPO (eps .2/.28, 2605.17770); no evidence for small models / ~1.5k prompts.
- TRL/Unsloth defaults (unverified): loss_type dapo, scale_rewards group, beta 0.0, num_generations 8. Dr.GRPO = constant-length normalizer, no std scaling.
- GRPO over SFT with overlap metrics: dialogue summarization Qwen2.5-3B GRPO ROUGE-L 49.2 vs 7B SFT 44.05 (2604.17188); spoken QA ROUGE-L 55.21 → 56.75 (2509.16990); counterexample: adding ROUGE to a semantic reward hurt (2509.13081). Gains small.
- Not found: prompt difficulty filtering, steps for 1.5k prompts, LoRA-vs-full lr comparison.
- Agent's starting recipe (own synthesis): G=8, lr ~1e-5 LoRA (1e-6 full), beta 0, dapo/dr_grpo loss, no std scaling, drop zero-variance groups, eps_high .28, 1–2 epochs.
