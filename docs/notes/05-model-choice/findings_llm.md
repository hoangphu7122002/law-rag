# Findings: instruct LLMs < 4B for Vietnamese legal RAG (2026-11-17)

Source: research subagent (web). [V] = verified on a page; [M] = from memory, check `config.json`.

| Model | Params | Ctx | License | Vietnamese evidence (self-reported unless noted) |
|---|---|---|---|---|
| Qwen3.5-2B (Mar 2026) | "2B" [V]; vision encoder inclusion unclear | 262K [V] | Apache-2.0 [V] | 201 languages; MMMLU 56.9 (non-thinking) / 63.1 (thinking); MMLU-ProX 52.3; INCLUDE 55.4; vLLM nightly [V] |
| Qwen3.5-0.8B | 0.8B [V] | 262K | Apache-2.0 | none |
| Qwen3.5-4B / 9B | excluded (≥ 4B) | | | |
| Qwen2.5-3B-Instruct | ~3.09B [M] | 32K | Qwen Research (non-commercial) [M] | base of the Vietnamese fine-tunes below |
| Qwen2.5-1.5B-Instruct | ~1.54B [M] | 32K | Apache-2.0 | weak Vietnamese expected |
| Qwen3-1.7B | ~2.03B incl. embeddings [M] | 32K | Apache-2.0 | MMMLU 57.0 (thinking) |
| Qwen3-0.6B | ~0.75B [M] | 32K | Apache-2.0 | likely too small to quote long passages |
| Arcee-VyLinh | ~3.09B [M] (one listing 3.4B, unverified) | 32K / 4096 (card revisions disagree) [V] | Qwen Research [M] | m-ArenaHard-vi (Claude judge): 95.4 % win vs PhoGPT-4B, 80 % vs Vistral-7B [V] |
| Vi-Qwen2-3B-RAG (AITeamVN) | "3B" [V] | 8192 (copied from 7B card) | Apache-2.0 [V] | card inconsistent; VMLU 56.04 belongs to the 7B [V]; trained to answer only from context, template `### Ngữ cảnh / Câu hỏi / Trả lời` |
| Qwen2.5-3B-legal-vn (bqbbao6) | ~3.09B [M] | | | QLoRA r16 legal RAG, no benchmarks, unvetted |
| Llama-3.2-3B / 1B | ~3.21B / ~1.24B [M] | 128K | Llama community | Vietnamese not officially supported [M] |
| Gemma-3-1B-it | ~1.0B [M] | 32K | Gemma terms | Gemma-2-2B vi fine-tune VMLU 42.35 (community) |
| Gemma 4 E2B | 2.3B effective, **5.1B total** [V] | 128K | | **excluded** |
| Phi-4-mini | ~3.84B [M] | 128K | MIT | weak Vietnamese [M] |
| SmolLM3-3B | ~3.08B [M] | 64K | Apache-2.0 | Vietnamese not official [M] |
| Cohere Tiny Aya | 3.35B [V] | | unchecked | no Vietnamese numbers |
| ViLegalQwen / ViLegalLM | not found | | | |

- VLSP2025-qwen3-4b-legal-pretrain: excluded (4B), but VLegal-Bench review says legal pretraining lets it match larger models → supports CPT direction.
- VLegal-Bench (arXiv 2512.14554): 22 tasks, 24 LLMs; per-model scores not extracted (try arxiv.org/html/2512.14554v4).
- No independent Vietnamese-legal faithfulness numbers for any < 4B model → dev-300 is the real test.

## Shortlist
1. **Qwen3.5-2B + LoRA** — newest, Apache-2.0, 262K ctx, best multilingual among Apache options. Risk: vLLM nightly, exact param count to confirm.
2. **Qwen2.5-3B-Instruct + LoRA** (optionally from Arcee-VyLinh) — most mature vLLM + Vietnamese ecosystem; non-commercial license OK for coursework.
3. **Qwen3-1.7B + LoRA** (cheap fallback) / **Vi-Qwen2-3B-RAG** zero-shot baseline.

LoRA on (context → reference) pairs likely matters more than the base choice. Run all on dev-300, temperature 0, pick by ROUGE-L.
