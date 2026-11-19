# Findings A — training-free edits (activations, weights, decoding)

Collected 2026-11-18 by a research subagent (search snippets + 1 page fetch). Items marked (unverified) come from
secondary summaries: check the paper before citing.

| Technique | Mechanism | Reported gains | Cost (vLLM / HF) | Risks | Verdict |
|---|---|---|---|---|---|
| Copy / pointer logit boost | add bias b to logits of tokens present in retrieved context | CoCoLex (copy-based legal decoding) > greedy, CAD, AdaCAD on correctness / faithfulness (snippet only, verify) | vLLM custom logits processor (~20 LOC), ~0 runtime; HF `LogitsProcessor` | b too high → repetition, wrong article; exempt EOS / format words | **TRY** |
| CAD (Shi et al.) | logits = (1+α)·logit(y\|ctx,q) − α·logit(y\|q); 2 passes | faithfulness / knowledge-conflict QA gains, LLaMA/OPT up to 30B (numbers unverified) | HF processor, ~2× compute; vLLM awkward (paired no-context request) | over-correction; fluency / format drop at high α | MAYBE (HF, α ≈ 0.3–1) |
| Adaptive CAD (AdaCAD, CoCoA, COIE) | per-step α from divergence / entropy | CoCoA up to +9.2 QA acc over AdaCAD, +2.5 summ / LFQA (self-reported) | CAD + few lines | more hyperparameters | MAYBE, only if CAD helps |
| Activation steering (CAA / RepE / ContextFocus) | mean-difference vector added at layer L (faithful-to-context vs prior answers) | ContextFocus: higher faithfulness on ConFiQA, Llama-3.1-8B, best layer 13, no fine-tune (+35–71 %, unverified); Shift = gated steering for RAG conflicts | HF hooks ~30 LOC + a few hundred contrast pairs; vLLM: patch model or bake vector offline | per-model layer / α; less stable on small models; fragile off-distribution; Vietnamese untested | MAYBE, after copy bias |
| Weight orthogonalization (Arditi) | W' = W − r rᵀ W removes a direction (refusal dir. in 13 models ≤ 72B) | removes refusal, small capability loss (overlap with instruction following, unverified) | offline HF edit, served by vLLM at 0 runtime | no refusal problem here; "prior reliance" direction unproven | MAYBE-LOW, only to bake a steering vector |
| Weight-space programming (blog) | frontier LLM reads / edits weights with forward-pass feedback | Qwen3.5-0.8B: loop direction removed in L15 / L19 MLPs (0.016 % weights): MATH-500 7.9→48.1, GSM8K 13.8→67.1, IFEval 21.0→62.3 with 50 forward passes | frontier-LLM harness + sandbox + eval passes | concept, not a method; removes obstacles, cannot add ability; one model | SKIP as a method |
| DoLa | final vs early-layer logit contrast | TruthfulQA +12–17 pts on LLaMA (open QA, not RAG) | HF `dola_layers`; not in vLLM | targets parametric facts, not copying | SKIP (optional 1-line HF ablation) |
| Contrastive decoding (expert − amateur) | subtract small-LM logits under plausibility mask | fluency in open generation, nothing RAG-specific | second model, HF only | no copy signal; hurts format | SKIP |
| ITI | shift attention heads along a probed "truthful" direction | Alpaca-7B TruthfulQA 32.5→65.1 (from memory) | HF hooks + probing | truthfulness ≠ copying | SKIP |
| Format steering vector | CAA on "Căn cứ … Như vậy" vs plain answers | no small-LLM numbers | HF hooks | prompt / few-shot / forced prefix is cheaper | SKIP |

## Notes for law-rag

- The metrics reward overlap with the quoted article → copy bias acts on that directly; CAD / steering only indirectly.
- The blog's Qwen3.5-0.8B gain (IFEval 21→62) came from removing loops / overlong output (token-limit hits
  93 % → 33 %). Check our generator for loops and truncation first; no-think mode + repetition penalty are cheap.
- vLLM custom logits processors exist but the API changed in V1; steering is not native (HF, or bake into weights).
- Baseline to beat: forced prefix "Căn cứ khoản X Điều Y … quy định như sau:" + top retrieved article span.

Suggested order: decoding hygiene (no-think, repetition penalty, forced prefix) → copy-bias b ∈ {0.5, 1, 2} →
CAD α ∈ {0.3, 0.5, 1} on dev → CAA vector from train (verbatim vs paraphrase) → bake the winner into weights.

## Sources

- CAD https://arxiv.org/abs/2305.14739 · CoCoA https://arxiv.org/html/2508.17670v1 · CoRect https://arxiv.org/pdf/2602.08221
- ContextFocus https://arxiv.org/html/2601.04131v1 · steering specificity https://arxiv.org/html/2602.06256v1 · Shift https://huggingface.co/papers/2606.27786.md
- DoLa https://arxiv.org/abs/2309.03883 · refusal direction https://arxiv.org/pdf/2406.11717 · false-refusal orthogonalization https://arxiv.org/pdf/2410.03415v2
- Weight-space programming https://rikiyat.github.io/blog/weight-space-programming/
- ITI arXiv 2306.03341, CAA arXiv 2312.06681 (from memory, not re-checked); CoCoLex seen only in a snippet.
