# Findings B — putting knowledge into weights vs RAG

Collected 2026-11-18 by a research subagent (search snippets, not full papers). (mem) = from memory, not verified.

**Bottom line:** no weight-space method beats RAG at our scale (~50k documents, ~1M articles). Published wins are on
single documents, corpora ≤ ~500k tokens, or ≤ ~10k facts. Verbatim quoting needs the exact text in context, so
retrieval stays the backbone; the useful methods train the reader to use retrieval better.

| Technique | Mechanism | Scale shown | Gain vs RAG / ICL | Compute | Verdict |
|---|---|---|---|---|---|
| RAFT (2403.10131) | SFT on question + oracle doc + distractors; answer quotes the doc | domain QA sets, Llama-2 7B | large gains over domain SFT and base + RAG (mem) | cheap LoRA SFT | **TRY first** |
| Knowledge Modules / deep context distillation (2503.08727) | per-doc LoRA matching a teacher's hidden states + logits | QuALITY, NarrativeQA; Phi-3 3B, Llama-3.1 8B | beats next-token LoRA; works alongside RAG | per-doc training (50k docs heavy) | MAYBE, hot statutes only |
| Cartridges / self-study (2506.06266) | trained KV cache via synthetic conversations + context distillation | one corpus 100k–484k tokens | matches ICL, 38.6× less memory, 26.4× throughput | GPU hours per corpus | MAYBE / skip (per-topic only) |
| Doc-to-LoRA (2602.15902) | hypernetwork turns one context into a LoRA in one pass | single long contexts; NIAH near-perfect at 5× window (Gemma-2-2b) | no RAG / 1M-doc evidence | hypernet once, < 1 s per doc | MAYBE-LOW; no verbatim guarantee |
| SPA (2603.22213) | simple prompted synthetic data for continued training | small corpora | claims > Active Reading | low | context: cheap synthetic-data baseline (feeds KD2 / RAFT) |
| EntiGraph (2409.07431) | entity graph → synthetic text → continued pretraining | 1.3M → 455M tokens, 8B | ~80 % of the in-context gain; still < RAG | days of GPU | SKIP |
| Active Reading (2508.09494) | model writes its own study material, then trains on it | up to 1T synthetic tokens, 8B | > paraphrase / QA augmentation | very high | SKIP |
| SEAL (2506.10943, mem) | model writes self-edits, RL on downstream reward | single passages, 7B | no-context SQuAD 33.5 → 47.0 (mem) | RL loop | SKIP |
| ROME / MEMIT / AlphaEdit | closed-form MLP edits | MEMIT ~10k facts; AlphaEdit ~2–3k sequential (mem) | triples only | min–hours | SKIP (law text is not triples) |
| Text-to-LoRA (2506.06105), Drag-and-Drop (2506.16406), Generative Adapter (2411.05877) | text / prompt → LoRA | task-level adapters | skills, not facts | hypernet training | SKIP |
| Test-time training on retrieved docs (mem) | gradient steps per query | per query | little evidence over in-context | sec–min per query | SKIP |
| Memory layers (2412.09764, mem) | sparse key-value memory in place of FFN | pretraining-time | dense match at 2× compute | needs pretraining | SKIP |

Plan: (1) RAFT on 2.2k train pairs + own-retriever top-k with the gold article, hard negatives, gold dropped in
20–30 % of samples, target = reference answer (verbatim spans); (2) synthetic QA over articles (SPA-style, decontaminated)
to grow RAFT data; (3) only if RAFT plateaus: Knowledge Modules / a Cartridge for the top 100–500 cited statutes.

Sources: Doc-to-LoRA https://arxiv.org/abs/2602.15902 (https://github.com/SakanaAI/doc-to-lora) ·
Drag-and-Drop https://arxiv.org/abs/2506.16406 · Cartridges https://arxiv.org/abs/2506.06266 ·
EntiGraph https://arxiv.org/abs/2409.07431 · Active Reading https://arxiv.org/abs/2508.09494 ·
SPA https://arxiv.org/abs/2603.22213 · Knowledge Modules https://arxiv.org/abs/2503.08727 ·
RAFT https://arxiv.org/abs/2403.10131 · MEMIT https://arxiv.org/abs/2210.07229 ·
from memory: Text-to-LoRA 2506.06105, SEAL 2506.10943, AlphaEdit 2410.02355, Memory Layers 2412.09764, Generative Adapter 2411.05877.
