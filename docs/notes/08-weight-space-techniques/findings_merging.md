# Findings C — model merging and weight averaging

Collected 2026-11-18 by a research subagent (4 searches). (prior) = from the original papers by memory.
No source tests merging for < 4B legal RAG scored by lexical overlap: transfer is untested; expect ~+0.5–2 pts.

**Verified:** across 6 merging methods × 4 base LLMs × 16 benchmarks, only plain task arithmetic gained reliably;
TIES tended to hurt (drifts from base, forgets) — https://arxiv.org/html/2511.21437v1, survey https://arxiv.org/abs/2603.09938.

| Method | Mechanism | Reported gains | Cost | Risk | Verdict |
|---|---|---|---|---|---|
| WiSE-FT (2109.01903, prior) | interpolate base and fine-tune, α ∈ [0, 1] | +4–6 OOD on ImageNet shifts, ~no in-dist. loss | CPU, α sweep | α too low → no domain gain | **TRY** (core recipe) |
| Task arithmetic (2212.04089, prior) | θ = base + λ (ft − base) | best reliable LLM merger in 2511.21437 | CPU | λ too high → forgetting | **TRY** (λ 1.2–1.5 extrapolation too) |
| Model soups (2203.05482, prior) | average same-init fine-tunes | +0.5–2 over best single | N runs | needs same init | **TRY** (seed / LR soups of LoRA) |
| Chat vector (2310.04799, 2412.11704) | add (chat − base) to a continually pretrained base | chat skills on a CPT model without SFT | CPU | needs matched base / chat pair | TRY only if we do CPT |
| TIES (2306.01708) / DARE (2311.03099) | trim + sign election / random drop + rescale | good on T5 / ViT, 7B+ math / code; TIES hurt LLMs | CPU | drift; small-scale unverified | MAYBE (≥ 3 models) |
| SLERP | spherical interpolation of 2 models | community use, no rigorous domain gains | CPU | per-layer t tuning | MAYBE |
| ChipAlign (2412.19819), CPT + merge (2406.14971) | geometry-aware domain + chat merge | better instruction following on 70B domain models | CPU | evidence at 70B | MAYBE |
| Evolutionary merge (2403.13187) | CMA-ES over per-layer weights | JP + math merge beat larger JP models | GPU eval loops | overfits search metric (dev 300) | MAYBE, late polish |
| LoRA merging (naive) | average A and B separately | underperforms unless LoRA-aware | cheap | averaging A, B separately is wrong | merge full ΔW = BA only |
| LoRA Soups CAT (2410.13025), LoraHub (2307.13269) | learned / searched mix of skill LoRAs | +43 % / +12 % on 2-skill math; ≈ ICL on BBH | small | needs several skill LoRAs | SKIP (one skill) |
| mergekit (https://github.com/arcee-ai/mergekit) | YAML: linear, slerp, task_arithmetic, ties, dare_ties; `mergekit-evolve` | — | free, CPU | same architecture | USE |

Risks here: Vi-Qwen2-3B-RAG's grounding is itself a fine-tune delta, so merging toward a generic chat model can dilute
it (always evaluate with retrieval in the loop); interpolation shifts answer length (check per α); α / λ sweeps on
300 dev items overfit (paired bootstrap, test held out).

Recipes, ranked: (1) after LoRA SFT, scale the delta α ∈ {0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5} (peft
`add_weighted_adapter` / merge-time scaling), pick on dev; (2) 3–5 LoRA SFTs (seeds, LR 1e-4 / 2e-4, ranks), average
full ΔW, or last-k checkpoint / EMA average; (3) chat vector only with CPT: θ = law_CPT + w (Vi-Qwen2-3B-RAG − Qwen2.5-3B),
w 0.5–1; (4) soup the teacher-distilled SFT with the direct SFT (same init); (5) optional mergekit `dare_ties` / `slerp`
/ `mergekit-evolve` on the L40S.
