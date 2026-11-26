# Research plan: retrieval sufficiency gate for Vietnamese legal RAG

**Main question.** After dense retrieval (top-50) + reranker (top-5), ~10% of questions have no gold article in the top-5 (those score ROUGE-L ~0.36 vs ~0.66 when gold is present; ceiling if fixed = +0.03). A cheap gate (logreg/GBM on reranker scores etc., AUROC ~0.83 on dev) can flag these. What is the best cheap design for (a) the gate, (b) the context-expansion step when flagged, (c) making a small (<4B) generator robust to expanded/noisy context? Metric: ROUGE-L. No big LLM at inference.

## Subtopics
1. **Gate / when to expand.** Retrieval-failure and sufficiency prediction: query performance prediction from reranker/dense score distributions, Adaptive-RAG (complexity classifier), CRAG (retrieval evaluator), Self-RAG, FLARE, "sufficient context" classifiers (Google 2024-25), calibration, threshold selection. Expected: which signals work, reported AUROCs, whether small classifiers match LLM judges.
2. **How to expand.** Query rewriting/HyDE/multi-query, entity (law-name)-constrained retrieval, citation/knowledge-graph expansion, neighbor / parent-child chunks, larger k, web fallback; legal-RAG and Vietnamese legal retrieval evidence (Zalo legal, BKAI legal retrieval, ALQAC, COLIEE), cost vs recall gain.
3. **Small generator robustness.** Training with distractors, dropped-gold + recovered contexts, refusal training, "RAG noise robustness" results (RAAT, RobustRAG, Self-RAG data, RAFT), effects on small models and ROUGE-style metrics; position/shuffle effects.

## Synthesis
Merge into a ranked recommendation (cheapest first) with expected gain, risk, and what to measure on our dev set; flag gaps/uncertainty.
