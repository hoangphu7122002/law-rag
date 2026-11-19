# Research plan — training the retriever (embedder + reranker) for law-rag, 2026-11-19

Question: how to fine-tune AITeamVN/Vietnamese_Embedding (bge-m3) and AITeamVN/Vietnamese_Reranker
(bge-reranker-v2-m3) on ~1.4k decontaminated train questions + ~2.5k Zalo pairs to lift Điều@1 (.61) / Điều@5 (.90)
toward Điều@20 (.97), without forgetting.

Subtopics (one agent each, 3–5 searches):
1. **Hard negatives + losses** — mining (top-k range, positive-aware / margin filtering of false negatives, NV-Retriever,
   multi-round), losses (InfoNCE / MNRL, CachedMNRL / GradCache, margin-MSE distillation reranker → bi-encoder,
   listwise / BCE / LambdaLoss for cross-encoders). → `findings_negatives_losses.md`
2. **Recipes for bge-m3 / bge-reranker-v2-m3** — FlagEmbedding / sentence-transformers fine-tune hyperparameters
   (lr, batch, group size, epochs, max length), data size needed, forgetting (LoRA vs full, mixing general data,
   model merging), AITeamVN / BKAI / ALQAC / Zalo 2021 Vietnamese legal recipes. → `findings_recipes.md`
3. **Synthetic queries** — LLM-generated questions per article (doc2query, InPars, Promptagator, GPL), consistency
   filtering, how many per document, gains on small in-domain sets. → `findings_synthetic.md`

Synthesis: `research_report.md` — recommended plan (reranker first), data, hyperparameters, eval on dev Điều@k.
