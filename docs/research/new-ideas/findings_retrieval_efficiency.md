# Retrieval / data / efficiency ideas (agent report, 2026-11-20; 4 searches, snippets only — every number UNVERIFIED unless noted)

## Ranked shortlist
1. **Oracle test for utility-labelled selection (A1).** Run the 2B generator once per candidate Điều (top-5/top-10) on ~2k dev questions, score ROUGE-L and cited Điều; compare the oracle-selected answer with the current top-1 answer. Oracle gap < 0.02 ROUGE-L → drop the whole family (reranker-in-the-loop, RL over passage selection, utility labels). Large gap → train the reranker on utility labels instead of gold labels. High value as a decision gate; a few thousand short generations.
2. **Cascade: greedy first, MBR only when uncertain.** Reuse existing MBR and greedy dev outputs; bucket the MBR gain by reranker top-1 margin or answer log-prob. If the gain is concentrated in the lowest ~30 % of margins the cascade keeps most of the MBR ROUGE-L at ~1.3x cost. No training.
3. **Adaptive k via reranker margin.** 1 Điều when the margin is large, 3-5 otherwise; sweep the threshold on dev; plot ROUGE-L vs mean prompt tokens. Shorter prompts (prefill dominates RAG latency, blog-level claim); may cut wrong-Điều citations from distractors (hypothesis, no source).
4. **Lossless speedups.** vLLM `quantization=fp8` (L40S supports FP8; blogs claim < 0.5 pt loss, ~33 % faster decode on a 7B/H100); prompt-lookup (n-gram) speculation fits RAG because answers copy Điều text, greedy stays lossless; static instructions first + `enable_prefix_caching` (retrieved Điều differ per query, gain likely small); AWQ INT4 riskier on a 2B model.
5. **Haiku listwise rerank check, then optional distillation.** Haiku-low listwise rerank of top-10 on ~300 dev questions where gold is in the top-10 but not top-1; measure Điều@1 vs the current reranker; if it wins, distil into a small student (LiT5-style). 300 calls fit the Haiku pacing rule.

## Not worth chasing now
RL/GRPO over passage selection (Rank-R1, ReasonRank: gains on reasoning-heavy BRIGHT; little headroom at ~90 % recall); synthetic QA per Điều (BKAI arXiv 2412.00657, NAACL 2025 short; embedder fine-tuning already null); hard-negative variants (arXiv 2505.19274: cross-encoder false-negative filtering not reliably helpful); curriculum / paraphrase augmentation (no measured legal gains found); answer-style augmentation (check output length vs gold length first).

## How to report latency
p50 / p95 per query split into retrieve, rerank, prefill (TTFT), decode; tokens in/out, batch or concurrency, GPU (L40S shared), warm/cold prefix cache, quantization; ROUGE-L from the same run alongside. Avoid throughput-only numbers.

## Sources
arxiv.org/abs/2312.02724 · cs.uwaterloo.ca/~jimmylin/publications/Tamber_Lin_ECIR2025.pdf · arxiv.org/abs/2503.06034 · arxiv.org/html/2508.07050v1 · arxiv.org/html/2412.00657v1 · preview.aclanthology.org/moar-dois/2025.naacl-short.12.pdf · arxiv.org/html/2505.19274v1 · arxiv.org/pdf/2502.17957 · arxiv.org/abs/2504.21015 · blog.premai.io/llm-latency-optimization-from-5s-to-500ms-2026 · pub.towardsai.net (vLLM optimization playbook for L40S) · youngju.dev (LLM inference optimization)

## Gaps
No direct evidence for generator-feedback rerankers in RAG; nothing on curriculum/paraphrase; efficiency numbers come from vendor/personal blogs → benchmark on our own 2B model and prompt mix.
