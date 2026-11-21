# Research plan: semantic chunking and multi-granularity chunking for legal RAG

Question: we only tried per-Điều chunks (2000 chars / 100 overlap). Would semantic chunking or different / multiple chunk lengths help retrieval and answers, esp. for article-structured statutes?

Subtopics
1. Chunking strategy comparisons: semantic vs fixed-size vs structure-aware (per-article/section); chunk-size sweeps; reported recall/answer gains and their size.
2. Multi-granularity: parent-child / small-to-big / hierarchical retrieval, late chunking, contextual retrieval (Anthropic), proposition chunking.
3. Legal-specific and Vietnamese evidence (legal chunking by article/clause, Vietnamese legal RAG chunk sizes).

Synthesis: map to our setting (articles already are the natural unit; gold is an article; Điều@5 .905) and say which ablation is worth running (e.g. 1000-char split of long articles, clause-level + parent article).
