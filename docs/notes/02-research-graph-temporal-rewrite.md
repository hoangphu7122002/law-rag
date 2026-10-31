# 02 — Research: Graph RAG with updates, temporal law versions, query rewrite, production (2026-10-31)

Web research, 2 rounds, ~21 pages (summary only; evidence JSON in the session scratchpad, not kept).

## Graph RAG — incremental insert / update / delete

| Option | Insert | Update / delete | Extraction LLM | Notes |
|---|---|---|---|---|
| LightRAG | incremental | delete by doc with KG regeneration (Aug 2025) | README floor ~Qwen3-30B-A3B → < 4B is below it | modes local/global/hybrid/naive/mix; Neo4j/Qdrant/Postgres backends |
| Neo4j + LlamaIndex PropertyGraphIndex | upsert by node id | `delete(ids)`; doc-level refresh unverified for graph stores | any | |
| neo4j-graphrag (python) | creates nodes | not documented → own MERGE / delete-by-doc Cypher | OpenAI/Vertex/Anthropic, custom LLMInterface | |
| Microsoft GraphRAG | `graphrag update` (standard/fast) | no delete | heaviest | Global / Local / DRIFT |
| Graphiti | incremental episodes | bi-temporal: invalidate, not delete | needs reliable JSON; small models fail | chat-memory oriented |
| FalkorDB GraphRAG SDK | hash-skip unchanged | `delete_document`, `apply_changes` without LLM | | vendor docs |

Take-away: with a < 4B model, build the graph **deterministically** (law structure + amendment metadata) in Neo4j with MERGE;
LightRAG / MS GraphRAG only as comparison arms on a subset with a larger extraction model.

## Temporal / versioned law

- Akoma Ntoso FRBR: fixed Work URI, one Expression per version date; `::date` = latest version on or before a date (Laws.Africa).
- Benchmarks: FiscalQA Pro (arXiv 2608.09393): static current-version RAG 2.7 % vs date-conditioned retriever 98.3 %
  (abstracts disagree, medium); German statutory QA (arXiv 2605.23497): as-of-date filtering helps all question types;
  TimelyRAG (arXiv 2609.11572): temporal reranking up to +28.6 % nDCG@10. SAT-Graph RAG (2505.00039): design, no metrics.
- VBHN (văn bản hợp nhất) is a consolidated convenience text; keep the amendment chain as the source of truth.
- Design: per article `effective_from`, `effective_to`; edges `amends`, `replaces`, `repeals`, `bases_on`; date filter at retrieval.

## Query rewrite (small-LLM fit)

Multi-query + RRF (safe default) · decomposition (helps multi-part, hurts simple lookups) · HyDE (arXiv 2212.10496;
risk: small model hallucinates) · RQ-RAG (2404.00610, +1.9 % with a fine-tuned 7B) · Rewrite-Retrieve-Read (2305.14283,
trained small rewriter). Most gain numbers are blog-level → measure ourselves.

## Production

Golden-set eval gate in CI (fail on drop); RAGAS is noisy, so thresholds from our own baseline. Qdrant (vectors + payload
filters on date / doc id) + Neo4j (relations) — our suggestion, not sourced.

## Sources

https://github.com/HKUDS/LightRAG · https://developers.llamaindex.ai/python/examples/property_graph/graph_store ·
https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html · https://microsoft.github.io/graphrag/cli/ ·
https://github.com/getzep/graphiti · https://docs.falkordb.com/graphrag/incremental-updates.md ·
https://developers.laws.africa/get-started/works-and-expressions · https://arxiv.org/abs/2506.07853 ·
https://arxiv.org/abs/2505.00039 · https://arxiv.org/abs/2608.09393 · https://arxiv.org/abs/2605.23497 ·
https://arxiv.org/abs/2609.11572 · https://arxiv.org/abs/2212.10496 · https://arxiv.org/abs/2404.00610 ·
https://arxiv.org/abs/2305.14283
