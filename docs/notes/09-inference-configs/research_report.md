# Inference configs for the < 4B reader (A3) — report, 2026-11-18

Sources: findings below (agents, 3–8 searches each, model cards fetched from Hugging Face). Items marked
"unverified" in the findings are not re-checked here. Reference answers: p50 ≈ 400 tokens, p90 ≈ 790, p99 ≈ 1,400
(Qwen tokenizer, train / dev) → `max_tokens` 768 cuts ~10 %; use 1,024 without thinking.

## Configs to sweep on dev (thinking on vs off)

| id | model | mode | temperature / top_p / top_k | penalties | max_tokens | note |
|---|---|---|---|---|---|---|
| G0 | Vi-Qwen2-3B-RAG | off (no thinking mode) | greedy (current A2e) | — | 1024 | baseline |
| G1 | Vi-Qwen2-3B-RAG | off | 0.1 / 0.8 / 20 (card example) | repetition 1.05 (generation_config) | 1024 | card prompt verbatim |
| Q0 | Qwen3.5-2B | off (default) | 1.0 / 1.0 / 20, min_p 0 (official non-thinking) | presence 2.0 | 1024 | official |
| Q1 | Qwen3.5-2B | off | 0.3 / 0.8 / 20 | presence 1.5 | 1024 | lower T for copying (our variant) |
| Q2 | Qwen3.5-2B | **on** | 1.0 / 0.95 / 20 (official thinking) | presence 1.5 | 3072, thinking cap 1024 | `--reasoning-parser qwen3`; 2B is "prone to thinking loops" (card) |
| Q3 | Qwen3.5-2B | **on** | 0.6 / 0.95 / 20 (datamate final, "precise" preset) | presence 0 | 3072, cap 1024 | same as datamate |
| R0 | Qwen3-1.7B | off | 0.7 / 0.8 / 20 | presence 1.5 if loops | 1024 | card: never greedy |
| R1 | Qwen3-1.7B | on | 0.6 / 0.95 / 20 | — | 3072, cap 1024 | |
| M  | best of the above | — | N = 8 samples (+ greedy) | — | — | MBR, ROUGE-L utility |

Excluded: Qwen3-4B-Instruct-2507 (4.0B total params > 4B rule), Qwen3.5-4B (4.66B). Llama-3.2-3B last: Vietnamese
is not an officially supported language.

## How
- Thinking switch per request: `chat_template_kwargs={"enable_thinking": true|false}`; serve with
  `--reasoning-parser qwen3` so only the text after `</think>` is scored. Qwen3.5 has no `/think` soft switch.
- Thinking cap (datamate pattern): generate the thought with `max_tokens = cap`; if no `</think>`, append it and
  continue for the answer.
- top_k / presence_penalty / min_p go in `extra_body` of the OpenAI client.
- MBR: `S[i][j] = rougeL(c_i, c_j)`, pick `argmax mean_{j≠i}`; candidates = N samples + greedy. Gains reported
  +1–7 ROUGE/BLEU in summarisation / data-to-text (Suzgun 2022, Chen 2023 USC); watch length (check output/ref
  length ratio on dev). N× generation cost, ROUGE matrix cheap on CPU.

## Order (dev, paired bootstrap vs G0, then test once)
G1 → Q0/Q1 → Q2/Q3 (thinking) → R0/R1 → MBR on the best → k = 10 on the best → test.
Prompt variants (reference style "Căn cứ … quy định như sau: …", few-shot) cross the best two configs.
