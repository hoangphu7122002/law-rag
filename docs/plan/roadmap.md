# Roadmap (draft — to settle after research)

| # | Step | GPU? |
|---|---|---|
| 0 | Data: unzip, clean RES, fixed split, data exploration notebook | no |
| 1 | Baseline re-run on `test` (≥ 100 q) with the exact metrics | small |
| 2 | Retrieval only: law-id hit@k for chunking × embedding × BM25/hybrid × reranker | small |
| 3 | Generation: instruct LLM < 4B, prompt with answer template, context packing | yes |
| 4 | Distinctive: Graph RAG over law structure and cross-references, and/or LoRA SFT on `train` | yes |
| 5 | Demo notebook / script, report, manual | no |
