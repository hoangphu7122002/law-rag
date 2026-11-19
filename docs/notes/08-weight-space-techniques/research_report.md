# Techniques beyond distillation — report (2026-11-18)

Trigger: "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL,
https://rikiyat.github.io/blog/weight-space-programming/): a frontier LLM reads / edits weights with no gradients.
On Qwen3.5-0.8B it removed a "looping" direction from 0.016 % of the weights (MATH-500 7.9 → 48.1). The authors call
it a concept, not a method, and show that edits remove obstacles but cannot add missing knowledge. The useful lesson
for us: **look for cheap failure modes (loops, truncation, not copying) before adding capacity.**

Findings: [inference edits](findings_inference_edits.md) · [knowledge injection](findings_knowledge_injection.md) ·
[merging](findings_merging.md).

## Ranked for law-rag (metrics reward verbatim overlap with the quoted article; LLM < 4B; RAG stays the backbone)

| # | Technique | Why it fits | Cost | Roadmap id |
|---|---|---|---|---|
| 1 | Decoding hygiene: loop / truncation check, no-think mode, repetition penalty, forced "Căn cứ …" prefix | the blog's whole gain was removing loops; free | minutes | W0 |
| 2 | Copy bias (logit boost for context tokens; CoCoLex-style legal copy decoding) | acts directly on the overlap metrics | ~20 LOC vLLM logits processor | A1 (existing) |
| 3 | RAFT (gold + distractors, answer quotes the article) | trains the < 4B reader to use retrieval; cheap LoRA | ~1 GPU-hour | T2 (existing) |
| 4 | WiSE-FT α-scaling + seed / checkpoint soups of the SFT / RAFT LoRA | free regularizer, keeps Vi-Qwen2 RAG skills; only task arithmetic is reliably good on LLMs | CPU | W1 |
| 5 | CAD / adaptive CAD (context vs no-context logit contrast) | more faithful to retrieved text | 2× decode, HF path | W2 |
| 6 | Activation steering (CAA / ContextFocus) toward context-faithful answers, baked into weights by orthogonalization if it wins | the weight-space idea in a reproducible form | HF hooks + few hundred pairs | W3 |
| 7 | Synthetic QA (SPA-style) over articles → more RAFT / KD data | widens coverage beyond 2.2k pairs | teacher time | KD2 (existing) |
| ◇ | Knowledge Modules / Cartridges for the top 100–500 cited statutes | complements RAG, but per-doc training | GPU hours | W4 |
| ◇ | Chat vector (only if continued pretraining P-1 is done), mergekit-evolve | late polish | CPU / GPU hours | W5 |

Skip: weight-space programming as a method (needs a frontier-LLM harness; no knowledge gain), knowledge editing
(ROME / MEMIT / AlphaEdit: triples, ≤ 10k facts), EntiGraph / Active Reading / SEAL (replace RAG, too costly at 1M
articles), Text-to-LoRA / Drag-and-Drop (skills, not facts), DoLa / ITI / contrastive decoding (no copy signal),
LoraHub / LoRA Soups (one skill only).

Caveats: most numbers come from search snippets; rows marked unverified / mem in the findings must be checked
before citing in the report. No source tests these on < 4B legal RAG with lexical metrics.
