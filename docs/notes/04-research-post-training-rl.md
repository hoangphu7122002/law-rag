# 04 — Research: post-training with RL (verl, GRPO, DPO) (2026-10-31)

Quick web research (3 notes + search snippets; summary only).

- **verl** (volcengine/verl, Apache-2.0): PPO, GRPO, GSPO, RLOO, REINFORCE++, DAPO, DrGRPO; FSDP/Megatron + vLLM/SGLang
  rollout; LoRA listed as multi-GPU; the Qwen2.5-3B GRPO+LoRA example uses 2–4 A100. No single-GPU evidence. Heavy setup (Ray).
- **TRL GRPOTrainer:** single GPU feasible with colocated vLLM (TRL ≥ 0.18). **Unsloth GRPO:** lightest VRAM (vendor claim),
  ≥ 300 steps recommended.
- **RL for RAG:** Search-R1 (arXiv 2503.09516): Qwen2.5-3B-instruct EM 0.336 (GRPO) vs 0.270 plain RAG, 8×H100, 500 steps;
  GRPO can reward-collapse. R1-Searcher, ZeroSearch: search agents, datacenter compute.
- **Overlap metrics as reward:** ROUGE-only reward → model copies whole articles (blog, low); GRPO length collapse on
  summarization fixed by length constraint (arXiv 2510.19325). Our BLEU brevity penalty + recall metrics both reward padding/copying.
- **DPO / ORPO / KTO:** DPO with LoRA has a free reference model (adapter off); ORPO replaces SFT; KTO needs only binary labels.

## Decision draft

1. SFT LoRA → RAFT (main gain, ~1–2 USD).
2. DPO (TRL, LoRA): chosen = reference, rejected = SFT model's worst sample (~1–3 USD).
3. Small GRPO (Unsloth or TRL + colocated vLLM) on the SFT checkpoint, 1 L4/A10G, 300–500 steps (~4–9 USD, time a 10-step trial first).
   Reward = 0.5 ROUGE-L + 0.3 cosine + 0.2 citation regex; penalties: copy of retrieved context above a threshold, length far from target.
4. verl: only as a learning comparison on 2 GPUs if budget remains.

Sources: https://github.com/volcengine/verl · https://huggingface.co/blog/vllm-colocate ·
https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo ·
https://arxiv.org/html/2503.09516 · https://ishanjmukherjee.github.io/reward-hacking-grpo · https://arxiv.org/html/2510.19325v1 ·
https://arxiv.org/pdf/2402.01306
