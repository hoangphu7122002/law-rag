# Plan: GRPO reward + config for law-rag (Qwen3.5-2B SFT -> GRPO)
Question: is reward = .6 rougeL + .2 token overlap + .2 citation - copy - length penalty sound, or do we need several configs?
Subtopics
1. reward_design: lexical-metric rewards (ROUGE/BLEU/cosine) for QA/summarization/RAG; hacking (copy, padding, length collapse); mixing weights; citation rewards; Vietnamese/legal examples.
2. algorithm_config: GRPO variants (DAPO dynamic sampling, Dr.GRPO, length norm), beta/KL (0 vs .02), group size, lr, steps, LoRA vs full, small-model/small-data (~1.5k prompts) recipes, difficulty filtering.
3. trl_practice: TRL GRPOTrainer settings (loss_type, scale_rewards, num_generations vs batch, vllm colocate, memory on 48GB with 2B), known issues with Qwen3.5/multimodal.
Synthesis: verdict on current reward, list of configs to A/B (small grid), risks.
