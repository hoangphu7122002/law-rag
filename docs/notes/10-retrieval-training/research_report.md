# Training the retriever (reranker + embedder) — what to do, 2026-11-19

Sources: `findings_negatives_losses.md`, `findings_recipes.md`, `findings_synthetic.md` (snippets; several items are
marked unverified there). Starting point (A2d test): Điều@1 .61, @5 .90, @20 .97 → the candidates already hold the
answer; the gap is ORDER → the reranker is the first lever, the embedder second.

## Plan

| Step | What | Settings (start) | Why |
|---|---|---|---|
| R1 | Reranker fine-tune, data = course train (decontam) + Zalo 2021 | group 1 pos + 7 hard neg; lr 1e-5–2e-5, 1–2 epochs, max len 512, best dev NDCG@10 checkpoint | Điều@20 .97 vs @1 .61: ordering problem |
| R1a | Loss A/B: BCE (pos_weight 7) vs listwise softmax over the group | same data | evidence conflicts (practitioner: BCE in-domain; MS MARCO study: listwise) |
| R1b | False-negative filter: drop negatives with dense score > 0.95 × positive's (NV-Retriever TopK-PercPos); drop same-document chunks without a heading | — | legal near-duplicates / amended articles are frequent false negatives |
| R1c | LM-Cocktail merge: average fine-tuned + base weights | weight 0.5 | BGE: −1 pt target, +1.5 pt general → guards the 1.4k-query overfit |
| R2 | Round-2 mining with the R1 reranker, retrain | positive-aware filter kept | ANCE-style refresh; no numbers found → only if R1 helps |
| E1 | Embedder (bge-m3) fine-tune: MNRL + 7 hard negatives, temp 0.02, lr 1e-5, eff. batch ≥ 32, 1–3 epochs; ablation Margin-MSE from the R1 reranker | re-embed 724k nodes (hours) | only if Điều@50 (candidate recall) is the limit after R1 |
| S1 | Synthetic questions per Điều (Haiku / Mistral, 4–8 real train shots rotated, 3 per Điều, T 0.8) → filter by reranker score + round trip, drop lexical copies → decontaminate vs dev + test | 10–50k pairs | BKAI: 500k synthetic pairs → big gains on Zalo/TVPL; widens beyond 1.4k + 2.5k |

Eval: `cli run <exp> --retrieval-only --set rerank.engine=local --set rerank.model=<dir>` on dev (Điều@1/3/5, MRR),
base model on the SAME engine as the reference; then generation with the best LLM config.

## Not doing
Condenser pre-training, multi-model ensembles, pseudo-labelling (Zalo notes 07), GradCache (memory only, no quality gain).

## Gaps
No official bge-reranker-v2-m3 fine-tune recipe found; multi-round mining and LambdaLoss have no numbers; AITeamVN cards
give no hyper-parameters.
