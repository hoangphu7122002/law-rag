# Findings: clause selection + steering (3 searches; snippet-level, numbers NOT verified)

## Selection / two-stage
- DaR (arXiv 2605.24454): decompose question into sub-questions, 1 core provision each; KoBLEX statutory QA; better retrieval+answer. 2B: decomposition needs strong LLM; use teacher offline. https://arxiv.org/html/2605.24454v1
- PARSER / KoBLEX (2509.01324): generate parametric provisions as query scaffold, then retrieve-rerank-select. 226 QA. Needs LLM for scaffold. https://arxiv.org/pdf/2509.01324v1
- Layered regs, lower-layer answers miss lexically (2604.06173). https://arxiv.org/pdf/2604.06173
- Version/in-force retrieval (AACL 2026): wrong-version evidence ~useless. https://aclanthology.org/2026.aacl-long.42.pdf
- Gap: no found paper on sentence-level select-then-generate for statutes. Idea: train small cross-encoder/extractor at clause granularity (cheap, 2B-feasible), feed only selected clauses to generator -> improves rougeL via quote overlap.

## Vietnamese
- VLQA (2507.19995): 3,129 expert triplets (q, articles, answer), 59,636 articles, 27 domains; BM25/dense baselines. https://arxiv.org/html/2507.19995v1
- VLSP 2025 DRiLL: hybrid retrieval + cross-encoder rerank, clause segmentation preprocessing. https://preview.aclanthology.org/anlp-naacl/2025.vlsp-1.20.pdf , .../2025.vlsp-1.19.pdf , .../2025.vlsp-1.23.pdf
- VLegal-Bench (2512.14554): 10,450 samples, article-clause-point structure, RAG tasks. https://arxiv.org/pdf/2512.14554v3
- ViHERMES (2602.07361): multihop healthcare regs, graph-aware wins. https://arxiv.org/html/2602.07361

## Steering / editing for length/style
- Stolfo et al. ICLR 2025 (2410.12877): instruction steering vectors from activation diffs; length+format; works on Gemma-2 2B, Phi-3. Practical for 2B: yes, cheap (vector add at one layer, no training). https://arxiv.org/html/2410.12877v2
- SAE vs activation-diff steering for sentence-length (2510.01246): diff-vectors work. https://arxiv.org/pdf/2510.01246
- Compositional steering tokens (2601.05062): steering tokens beat act. steering on multi-constraint incl. length; combine with instructions. Trainable, 2B ok. https://arxiv.org/html/2601.05062v1
- 2605.23040: small models give diluted style steering; query-level interventions more stable. https://arxiv.org/html/2605.23040v1
- Style heads (2603.13249): few attention heads control style (7-8B). https://arxiv.org/pdf/2603.13249v1
- Fine-tuning vs steering (EMNLP 2026 main.567): brevity test; FT may hide/undo steering. https://aclanthology.org/2026.emnlp-main.567.pdf

## Caveat
Search snippets only; no gains numbers verified. Read papers before citing.
