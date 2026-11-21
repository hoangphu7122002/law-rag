# Multi-granularity retrieval: findings

Date: 2026-11-21. 4 web searches + 2 page fetches.
Legend: [READ] = fetched page, answered via fetch tool (summarised by small model, quotes may be paraphrased). [SNIPPET] = only search-result snippet/summary, not opened. All gains are authors' own numbers unless noted.

## 1. Contextual Retrieval (Anthropic)
Source: https://www.anthropic.com/engineering/contextual-retrieval [READ]
- Prepend 50-100 token chunk-specific context (LLM-generated) before embedding and BM25.
- Top-20 failure rate (1 - recall@20), baseline 5.7%:
  - contextual embeddings: 3.7% (-35%)
  - + contextual BM25: 2.9% (-49%)
  - + reranking: 1.9% (-67%)
- Cost: "$1.02 per million document tokens" one-time (800-token chunks, 8k-token docs, 100 ctx tokens/chunk); prompt caching cuts cost up to 90%, latency >2x.
- Datasets: codebases, fiction, ArXiv, science papers (long docs). No answer-quality metric reported in the post (retrieval only).
- Quote: if KB < 200,000 tokens, "you can just include the entire knowledge base in the prompt".
- Critique [SNIPPET]: third-party (Medium, almond-ai) says gains not additive on top of hybrid+rerank; realistic ~35%. https://medium.com/almond-ai/responding-to-anthropics-contextual-retrieval-for-rag-apps-why-context-is-not-all-you-need-af503985aa55

## 2. Proposition indexing (Dense X Retrieval, Chen et al., EMNLP 2024)
Source: https://arxiv.org/abs/2312.06648 [SNIPPET only]
- Index atomic, self-contained propositions vs passages/sentences.
- Avg Recall@5 on 5 datasets: +12.0 SimCSE, +9.3 Contriever (35.0% / 22.5% relative), unsupervised retrievers.
- Recall@20: +10.1 unsupervised; supervised +2.2 (v1) / +2.7 (later version).
- Authors: smaller gain for supervised retrievers (trained on query-passage pairs).
- Cost not seen: needs a propositionizer LLM pass over corpus and ~more index entries. UNVERIFIED.
- Corpus is Wikipedia passages (my recollection, not in snippet) - unverified.

## 3. Late chunking (Jina)
Sources: https://arxiv.org/abs/2409.04701 ; https://github.com/jina-ai/late-chunking ; https://jina.ai/news/late-chunking-in-long-context-embedding-models/ [SNIPPET only]
- Embed whole doc with long-context model, then chunk + mean-pool token embeddings. No training needed.
- BeIR: avg relative nDCG@10 gain 2.70%-3.63% over naive chunking.
- README: "average length of the documents correlates with greater improvement". Per-dataset (secondary, unverified): SciFact +1.9, TRECCOVID +1.34, NFCorpus +6.52.
- Caveat: needs long-context embedding model; models still better on short text. No independent replication seen.

## 4. Hierarchical / auto-merging (HiChunk + Auto-Merge)
Source: https://arxiv.org/html/2509.11552v3 (ACL 2026) [READ]
- Auto-Merge: merge children into parent if >=2 children retrieved, combined length > adaptive threshold, budget fits.
- Qwen3-32B, FC200 (fixed 200) -> HC200+AM:
  - HiCBench T1 evidence recall 74.06 -> 81.03; Rouge 35.70 -> 37.29
  - HiCBench T2 recall 72.95 -> 80.65; Rouge 37.17 -> 37.37
  - Qasper recall 84.32 -> 87.85; F1 46.49 -> 47.25
  - GutenQA recall 64.50 -> 65.53; Rouge 44.86 -> 44.94 (tiny)
  - OHRBench: "very close" across methods
  - LongBench avg: 46.33 (FC200) vs 46.92 (HC200+AM) vs 47.43 (LumberChunker)
- Depends on evidence density, not just length: "evidence sparsity" explains null on GutenQA/OHRBench. Level-1-only merge degrades; recall rises to level 3 then plateaus.
- Doc lengths: Qasper ~4.2k words, HiCBench ~8.5k, OHRBench ~26.8k, GutenQA ~146.5k.
- Chunker time s/doc: semantic(SC) 0.49-16.5, LumberChunker(LC) 5.5-132, HiChunk(HC) 1.5-60. Gain recall is retrieval-side; answer-quality (Rouge) gains much smaller than recall gains.
- No head-to-head vs LlamaIndex parent-child/small-to-big found [SNIPPET search]. Small-to-big/parent-child: no quantitative source found.

## Takeaways for law-rag
- Recall gains are large for dense retrievers on long unstructured docs; shrink with supervised/hybrid+rerank; answer metrics (Rouge) move far less than recall (matches project rule: gains must reach rougeL).
- If corpus is already short structured units (articles/clauses), late chunking and contextual retrieval have less to fix; contextual still helps when a short unit lacks context (e.g. "this article" refs). Parent-child merge is useful only if units have hierarchy (law > chapter > article > clause). Propositions help mostly weak/unsupervised retrievers.
- Costs: contextual = one LLM call per chunk (~$1/M tok w/ caching); propositions = LLM pass + larger index; late chunking = long-context embedder, no LLM; auto-merge = index-time hierarchy, cheap at query time.
- Gaps: no Dense X / Jina full-text read; no small-to-big numbers; no legal-domain evidence found.
