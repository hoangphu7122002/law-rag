# Deep-read findings on the sufficiency gate (3 subagents, full papers where noted; numbers as reported by the agents, not independently cross-checked)

## Google "Sufficient Context" (arXiv 2411.06037, full PDF read)
- Label = context contains all info for a plausible definitive answer, judged from (Q, C) only. 115 human-labelled instances. Autoraters (Table 1, n=115): Gemini 1.5 Pro 1-shot F1 .935 / acc .930; 0-shot .878/.870; FLAMe 24B .892/.878; TRUE-NLI T5-11B .818/.826 (needs gold answer); contains-GT .810/.809. No model below 11B tested as labeller; no AUROC reported.
- Selective generation: logistic regression on 2 features (sufficiency label + self-rated confidence); +2-10% correct among answered; no gain where accuracy is low (Gemma 27B on Musique, coefficient 0). Models still answer correctly 35-62% with insufficient context, so abstaining on every insufficient case is too blunt. FT with "idk" targets did not raise abstention for Mistral-7B (Table 3).
- Base rate insufficient: 55.4% over standard datasets (intro), HotpotQA/Musique ~45-55% sufficient depending on context cap.

## Learned gates from retriever signals
- Closest to our problem: Veselý et al., arXiv 2609.11646 "Your Retriever Already Knows: Distribution-Shape QPP for RAG Retrieval Sufficiency". Label Hit@k; 24 features (15 score-distribution-shape: top1-top10 gap, decay slope, bimodal gap, top-50 skewness; 5 query-surface; 4 global); LR/Ridge/MLP. Dense scores only. ViDoRe AUROC .856 (24 feat), .835 (18 classic), .593 (max score alone), LLM judge .649, hybrid .863; SUJB Hit@5 .893. 2 ms/query vs 6.3 s LLM judge. Leave-one-domain-out falls to .706: train in-domain. No downstream answer-quality measure. (Dated Sept 2026; not cross-checked.)
- Practitioner post (dev.to, unverified): within-query features on reranker logits (top1, margin, z-score of top1, logsumexp gap) beat raw top1; no numbers.
- QPP for RAG (Tian et al. 2601.14546): classic NQC near useless on neural rankers (Spearman .07-.20); combining with context perplexity etc. up to ~.25 (retrieval utility), ~.40 (answer quality).
- Adaptive-RAG: T5-Large query-only, 54.5% 3-way accuracy; oracle classifier F1 56.28 vs 46.94 -> headroom. Probing-RAG, SUGAR, DRAGIN, SeaKR, Rowen need LLM internals or sampling -> not usable for our frozen small pipeline.
- Gap: no result using cross-encoder reranker scores or dense-vs-reranker agreement as features.

## Small judges and legal
- ARES (2311.09476): DeBERTa-v3-Large 435M, context-relevance acc 67.8-92.3% by dataset; needs ~150 human labels for PPI validation; degrades below ~100.
- CRAG evaluator: T5-large (unverified size), scalar relevance in [-1,1] with thresholds; 2026 reproduction says it relies on entity alignment and transfers poorly across domains.
- Cheap labels: gold article present in top-k; spot-check ~150 by hand ("gold present" != "answerable": duplicate/superseded articles).
- Legal: statutory-context incompleteness drives hallucination (2604.06173; numbers unreadable); Legal RAG Bench 2603.01710 says many hallucinations are retrieval failures; no Vietnamese/ALQAC/COLIEE paper with an explicit "no relevant provision" classifier found.
- Evaluation: AUROC + ECE + selective accuracy vs coverage; false flag costs coverage/answer quality, missed flag costs hallucination; keep test untouched.
