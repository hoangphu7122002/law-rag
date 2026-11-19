# Research plan — inference configs for the < 4B reader (A3), 2026-11-18

Question: per candidate reader LLM, which decoding config (thinking on / off) should A3 test so answers overlap the
reference (BLEU-1, ROUGE-L, cosine)? References quote the statute article verbatim ("Căn cứ … quy định như sau: …").

Subtopics (one agent each, 3–5 searches):

1. **Qwen3.5 small (2B, 0.8B) + Qwen3 (1.7B, 4B-Instruct-2507 size)** — official sampling for thinking / non-thinking,
   `enable_thinking` + `--reasoning-parser` in vLLM, thinking budget / max_tokens, presence_penalty for loops,
   known issues (endless thinking, repetition, Vietnamese). → `findings_qwen.md`
2. **Non-thinking readers: AITeamVN/Vi-Qwen2-3B-RAG, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct** — model-card
   generation config, prompt format the RAG model was trained with, repetition settings, Vietnamese quality.
   → `findings_readers.md`
3. **MBR / self-consistency for free-text generation** — sample count, temperatures, utility (ROUGE / BLEU /
   embedding), reported gains on summarisation / QA, cost. → `findings_mbr.md`

Synthesis: `research_report.md` — one table of A3 configs (model × mode × sampling × max_tokens), the dev sweep order,
and caveats. Constraint reminders: LLM < 4B params, vLLM 0.30 on L40S, metrics never changed.
