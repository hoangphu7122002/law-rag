# Semantic chunking: is it worth it for Vietnamese law? (2026-11-18)

## Evidence
- Vectara, NAACL 2025 (arXiv 2410.13070): fixed-size beat semantic chunking on document retrieval, evidence retrieval
  and answer generation with realistic documents; the embedding model mattered more than the chunker.
- arXiv 2603.06976 (36 strategies, 6 domains, 5 embedders): structure-aware / adaptive chunking beat fixed-size and was
  more robust; best = paragraph-group chunking; costlier.
- Chunking German Legal Code (arXiv 2605.19806): section / subsection chunking had the highest recall; semantic
  clustering mid-tier; LumberChunker (LLM boundaries) 0.39-0.44 and fixed windows 0.30-0.40 worse. Pooling provisions
  blurs legal details.
- Kim & Choi, ICAIIC 2026: split by article, then by paragraph if too long = MRR 0.737, matching large-context (0.736).
- Semantic chunking end-to-end reported last in a 2026 vendor benchmark (fragments ~43 tokens); not peer reviewed.

## Models / tools if we try it
| Method | Tool | Vietnamese? | Cost |
|---|---|---|---|
| Embedding-breakpoint semantic | LlamaIndex `SemanticSplitterNodeParser` + Vietnamese_Embedding | yes | embed every sentence (millions) |
| Learned sentence/paragraph segmenter | SaT (wtpsplit), multilingual | yes | small model, cheap |
| Neural chunker | Chonky (DistilBERT, English BookCorpus) | no | - |
| Late chunking | Jina paper (arXiv 2409.04701); Chonkie `LateChunker`; needs long-context embedder (Vietnamese_Embedding 2048, bge-m3 8192) | yes via bge-m3 | one long-doc forward per file |
| LLM boundaries | LumberChunker | weak with < 4B | slow |

## Decision
1. C3 Điều/Khoản (`chunk.kind: dieu`, already coded) first: legal benchmarks favour structure, and answers cite "khoản .. Điều ..".
2. Then C6 as an ablation only: `kind: semantic` (LlamaIndex splitter, breakpoint percentile 95) on top of the best embedder.
3. Late chunking is a retrieval-side variant (same chunks, better embeddings): try with bge-m3 only if time remains.

## Vietnamese-specific (asked 2026-11-18)
No public Vietnamese semantic-chunking model was found. Build it from Vietnamese parts:
- **Sentence / unit splitter:** law text already has units per line (Điều, khoản "1.", điểm "a)") -> split on those lines
  first; fallback `underthesea.sent_tokenize` (Vietnamese) or SaT `sat-3l-sm` (wtpsplit, `vi` supported, no VN benchmark).
- **Embedder for breakpoints:** AITeamVN/Vietnamese_Embedding (bge-m3 FT, no word segmentation needed).
  PhoBERT-based models (vietnamese-sbert, bkai) expect RDRSegmenter word-segmented input (arXiv 2409.13699).
- **Wiring:** LlamaIndex `SemanticSplitterNodeParser(embed_model=Vietnamese_Embedding, sentence_splitter=<vi splitter>,
  breakpoint_percentile_threshold=95)`, then cap at ~2000 chars.
- VN legal systems in the literature chunk by section/article (arXiv 2409.13699; VLSP 2025 hierarchical chunking);
  SBV-LawGraph uses semantic chunking + hybrid + rerank (no ablation found).
