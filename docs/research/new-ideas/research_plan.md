# Research plan — new ideas to improve the small Vietnamese legal RAG + a study path (2026-11-20)

Main question: which NEW techniques (not yet tried here) could raise ROUGE-L / cut latency for a <4B Vietnamese legal RAG, and what should the user read/do to learn them while implementing?

Already tried / in progress (excluded): MBR decoding, SFT with shots + refusal rows, GRPO reward variants, on-policy distillation plan (Qwen3.5-9B teacher), ReAct trajectory SFT, embedder fine-tuning (no gain), chunk context lines (no gain), kNN prior from train questions, weight merging (soup / task arithmetic), query-rewrite survey.

Subtopics (3 parallel agents, 3–5 searches each):
1. **Generator-side training and decoding** — preference tuning (DPO / SimPO / KTO / ORPO) on candidate answers ranked by ROUGE-L, best-of-n / MBR distillation into greedy, self-consistency distillation, length/format control, constrained or template-guided decoding for the "Căn cứ Điều X … quy định …" style, citation-aware rewards, test-time compute scaling for small models; evidence for 1–4B models.
2. **Retrieval / data / efficiency side** — reranker-in-the-loop training or RL, listwise rerank with small LLMs, curriculum and data augmentation (question paraphrase, hard negatives), long-context packing, speculative decoding, quantisation (AWQ/GPTQ/FP8), prefix caching, to cut latency without losing ROUGE-L.
3. **Study path** — best learning resources in order: official docs and notebooks (TRL, PEFT, vLLM, LlamaIndex, Hugging Face course/smol-course), key papers to read first for each technique (LoRA, DPO, GRPO/DeepSeekMath, R1, on-policy distillation, MBR, RAG evaluation), good blog posts/tutorials; for each, what to read first, how long, and what to implement alongside (map to this repo).

Synthesis: docs/research/new-ideas/research_report.md — ranked new ideas (expected value, cheapest test, cost) + a study path table (topic → read → do in repo). Claims read only from snippets are marked UNVERIFIED.
