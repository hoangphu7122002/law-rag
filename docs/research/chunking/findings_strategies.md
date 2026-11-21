# Chunking strategy findings (fixed vs semantic vs structure-aware; size sweeps)

Date: 2026-11-21. 4 web searches + 6 page fetches.
Legend: [READ] = page fetched and read (via WebFetch summarizer, numbers not independently re-verified against the page); [SNIPPET] = only seen in search snippet.

## 1. Is Semantic Chunking Worth the Computational Cost? (Qu, Tu, Bao; Vectara; NAACL Findings 2025) [READ]
- https://arxiv.org/html/2410.13070v1 ; https://aclanthology.org/2025.findings-naacl.114 [SNIPPET]; blog https://vectara.com/blog/is-semantic-chunking-worth-the-computational-cost [SNIPPET]
- Compares fixed-size, breakpoint-based semantic, clustering-based semantic. Embedder stella_en_1.5B_v5; doc retrieval F1@5 on 100 queries/dataset; answers via gpt-4o-mini top-5.
- Quotes: "these benefits were highly context-dependent and did not consistently justify the additional computational cost"; "fixed-size chunking remains a more efficient and reliable choice for practical RAG applications".
- Document retrieval F1@5 (fixed / breakpoint / cluster), * = artificially stitched long docs:
  - Miracl* 69.45 / 81.89 / 67.35; NQ* 43.79 / 63.93 / 41.01 (breakpoint wins big on stitched topic-shift docs)
  - Scidocs* 16.82/17.60/19.87; Scifact* 35.27/36.27/35.70; BioASQ* 61.86/61.87/62.49; NFCorpus* 21.36/21.07/22.12 (~tie)
  - Natural docs: HotpotQA 90.59/87.37/84.79; MSMARCO 93.58/92.23/93.18; ConditionalQA 68.11/64.44/65.94; Qasper 90.99/89.27/90.77 (fixed wins)
- Evidence retrieval F1@5: ExpertQA 47.11/47.08/46.87; DelucionQA 43.05/43.24/43.36; TechQA 28.98/28.49/27.96; ConditionalQA 18.23/19.83/19.14; Qasper 8.66/8.16/8.50 (differences <~1.6 pts).
- Answer generation BERTScore: identical to within 0.01 (e.g. ExpertQA 0.65 all three; ConditionalQA 0.42/0.43/0.43).
- Caveat: chunk sizes of chosen configs not reported; stitched-doc gains are an artifact of synthetic topic boundaries.

## 2. Chroma: Evaluating Chunking (Smith & Troynikov, 2024) [READ]
- https://www.trychroma.com/research/evaluating-chunking
- Token-level metrics: recall, precision, IoU. text-embedding-3-large, n=5 retrieved chunks, all corpora (%):
  - Recursive 200 (no overlap): R 88.1 / P 7.0 / IoU 6.9
  - Recursive 400: R 89.5 / P 3.6 / IoU 3.6
  - Recursive 400 overlap 200: R 88.1 / P 3.3
  - TokenText 200: 87.0 / 5.2; TokenText 400: 89.2 / 2.7
  - Kamradt semantic (~660 tok): R 83.6 / P 1.5 (worst)
  - KamradtModified 300: 87.1 / 2.1
  - ClusterSemantic 200: R 87.3 / P 8.0 / IoU 8.0 (best precision)
  - ClusterSemantic 400: R 91.3 / P 4.5
  - LLM chunker (GPT-4o, ~240 tok): R 91.9 / P 3.9 (best recall, costly)
- Smaller chunks -> higher precision/IoU. Overlap mostly hurts IoU (TokenText 800/400 IoU 1.4 vs 2.7 at 400/0); can help recall for small chunks with weak embedder (MiniLM TokenText 250: 82.4 w/ 125 overlap vs 77.1 none).
- Takeaway quote: "reducing chunk overlap improves IoU scores"; Recursive 200/0 "performs well". Cheap heuristic ~ within 1-3 recall pts of semantic/LLM.
- Original Kamradt semantic is worst; cluster-semantic and LLM chunkers help modestly.

## 3. NVIDIA: Finding the Best Chunking Strategy for Accurate AI Responses [READ]
- https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/
- End-to-end RAG accuracy, 5 datasets. Page-level: avg 0.648, std 0.107 (highest, lowest variance). Token-based sizes: avg 0.603-0.645.
- Per-dataset: FinanceBench page 0.566, 1024tok 0.579, 2048tok 0.506; Earnings 512tok 0.681, 1024 0.663, 2048 0.651; KG-RAG page 0.520, 128tok 0.421 (worst); RAGBattlePacket page 0.790, 128tok 0.749, 1024 0.804, 2048 0.749.
- Query type: factoid -> 256-512 tokens; complex/analytical -> 1024 or page-level. "Extreme chunk sizes show diminishing returns". Section-level slightly above page-level on FinanceBench only; page boundaries "typically provide more coherent and effective chunks".
- NeMo Retriever docs default: 1024 tokens, 150 overlap; recommend page-level [SNIPPET] https://docs.nvidia.com/nemo/retriever/extraction/chunking

## 4. Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity (arXiv 2603.06976, 2026) [READ]
- https://arxiv.org/html/2603.06976 ; 36 strategies, 6 UltraDomain domains (incl. Legal), 5 embedders, LLM-judge relevance (Mixtral-8x22B), single run.
- Mean nDCG@5: Paragraph Group Chunking 0.459 (best), Dynamic token-size 0.441, LLM boundary 0.415; Fixed Character 0.244 (worst, P@1 ~2-3%). PGC P@1 10.4x fixed-char.
- Content-aware "significantly improves retrieval effectiveness over naive fixed-length splitting".
- Semantic mixed: legal Semantic Similarity Threshold Hit@5 ~54% but P@1 ~17%. Legal: PGC Hit@5 ~78%, P@1 ~35%.
- Efficiency: topic-based chunking 133.8 s vs PGC 6.3 s. "no single chunking strategy is universally optimal".
- Note: "fixed" baseline here is character-level, no sentence/paragraph awareness - weaker than recursive splitter baselines elsewhere.

## 5. Rethinking Chunk Size for Long-Document Retrieval (arXiv 2505.21700v2) [READ]
- https://arxiv.org/html/2505.21700v2 ; fixed token chunks 64-1024, no overlap, Stella and Snowflake-arctic-l-v2.
- Recall@1 (Stella/Snowflake): SQuAD* 64tok 64.2/60.9 -> 1024tok 38.6/42.9 (small better); TechQA* 64 4.9/4.5 -> 1024 61.9/71.5; NQ 64 17.2/15.0 -> 1024 34.9/47.7; NarrativeQA 4.2/3.4 -> 10.7/10.4; COVID-QA* mixed (Stella best at 64).
- Abstract: "smaller chunks (64-128 tokens) are optimal for datasets with concise, fact-based answers"; large chunks for long/dispersed answers. Optimum depends on embedder.
- Paper text has internal inconsistencies vs its tables (summarizer flagged); trust the table.

## 6. Legal / structure-aware [SNIPPET only]
- German Civil Code study: chunking aligned to section/subsection structure gave highest recall; more complex methods overriding structure did worse (cited via arXiv 2603.25333? unverified id; found via search 4, not fetched).
- NitiBench (Thai legal QA, https://arxiv.org/pdf/2502.10868): hierarchy-aware chunking preserves self-contained legal rules; authors note end-to-end experiment did not fully show the benefit.
- Also seen, not read: "When Is Complex Chunking Worth It?" https://arxiv.org/pdf/2608.16586 ; DLR/IEEE 90-config token-IoU benchmark (best: sentence splitter 512 window/200 overlap) https://elib.dlr.de/221921/1/COINS_CAMERA_READY_IEEE_APPROVED.pdf ; Splunk summary https://www.splunk.com/en_us/blog/artificial-intelligence/chunking-techniques.html

## Implications for law-rag
1. Semantic chunking: no consistent gain on natural docs (Vectara), worst/mid in Chroma; gains only on stitched/topic-shift docs. Skip for legal text with explicit structure.
2. Structure-aware (article/clause/section) is the natural baseline for statutes; evidence (NVIDIA page/section, legal snippets) suggests >= fixed-size, with lower variance. Test it first.
3. Size sweep to run: 128/256/512/1024 tokens + whole-article; expect factoid -> 256-512, analytical/multi-article -> larger; optimum embedder-dependent.
4. Use recursive splitter with ~0 overlap as the cheap fixed baseline; overlap rarely pays (Chroma, NVIDIA 150 overlap default is the exception).
5. Report token-level precision/IoU as well as recall; recall differences among chunkers are small (1-4 pts).
