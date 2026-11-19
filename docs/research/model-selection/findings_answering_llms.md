# Findings: answering LLMs < 4B (agent summary; few Vietnamese scores verifiable in 7 calls; (U) = unverified)
- No leaderboard with VMLU / ViMMLU / SeaExam / VieQA rows for these small models turned up. No evidence for any model on Vietnamese legal QA or RAG faithfulness.
- Qwen3.5-2B: card says "2B" (our own safetensors count: 2.27B, in roadmap), vision encoder included; Apache-2.0; multilingual only (thinking mode): MMMLU 63.1, MMLU-ProX 52.3, INCLUDE 55.4.
- Qwen3-4B: 4.0B total / 3.6B non-embedding (card) → sits AT the limit (roadmap already excludes it as 4.0B); Qwen3-1.7B ~1.7B (U); Qwen2.5-3B ~3.09B, Qwen-Research licence non-commercial (from memory, U).
- Gemma 3 1B (Global-MMLU-Lite 24.9, MGSM 2.04) / 4B (57.0, 34.7); the 4B figure includes the vision tower, text-only count not verified; Gemma terms.
- Llama-3.2-3B ~3.2B (U), Phi-4-mini ~3.8B MIT (U), SmolLM3-3B ~3B (U): not checked for Vietnamese.
- Vi-Qwen2-3B-RAG (AITeamVN): card shows VMLU only for the 7B sibling (avg 56.04); card inconsistent (names Qwen2-7B as base): 3B claim and RAG results unverified. Sailor2-3B: metadata says 4B (conflicts with name) → risky for the limit. ViLegalQwen / Vistral / VyLinh / SeaLLM: not checked.
- VMLU ACL 2025 paper exists (aclanthology.org/2025.acl-long.563), no small-model rows found; ViExam / VMMU (arXiv 2508.13680) multimodal only.
- Agent's next step: run our own evaluation on dev_clean / dev_strict with the same retrieval (Qwen3-4B only if its exact safetensors total < 4.0B).
Sources: huggingface.co/AITeamVN/Vi-Qwen2-3B-RAG, Qwen/Qwen3-4B, Qwen/Qwen3.5-2B, google/gemma-3-4b-it, sail/Sailor2-3B, arxiv.org/abs/2508.13680.
