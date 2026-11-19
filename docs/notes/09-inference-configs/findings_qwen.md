# Findings — Qwen3.5 / Qwen3 small models (agent, 2026-11-18; model cards + vLLM docs)

- **Qwen3.5-2B / 0.8B** (<4B; multimodal): thinking (text) T 1.0, top_p 0.95, top_k 20, min_p 0, presence 1.5;
  non-thinking (default) T 1.0, top_p 1.0, top_k 20, presence 2.0; "precise" thinking T 0.6 / 0.95 / 20, presence 0.
  Toggle `chat_template_kwargs.enable_thinking`; no /think soft switch; vLLM `--reasoning-parser qwen3`,
  server default via `--default-chat-template-kwargs '{"enable_thinking": false}'`. Card: 2B / 0.8B in thinking mode
  "more prone to entering thinking loops". Context 262k. Vietnamese quality unverified.
- **Qwen3-1.7B** (1.7B): thinking 0.6 / 0.95 / 20; non-thinking 0.7 / 0.8 / 20; presence up to 1.5 for repetition;
  "DO NOT use greedy decoding". `/think`, `/no_think` work when enable_thinking=True. Context 32k.
- **Qwen3-4B-Instruct-2507**: 4.0B total (3.6B non-embedding) → over the < 4B rule; non-thinking only, 0.7 / 0.8 / 20.
- No official thinking-budget API; cap = max_tokens then force `</think>` (unverified, datamate does this).

Sources: https://huggingface.co/Qwen/Qwen3.5-2B · https://huggingface.co/Qwen/Qwen3.5-0.8B ·
https://huggingface.co/Qwen/Qwen3-1.7B · https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ·
https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html ·
https://docs.vllm.ai/api/vllm/reasoning/qwen3_reasoning_parser.html
