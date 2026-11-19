# Query-side techniques for law-rag: findings

Scope: web research (11 searches, snippet/abstract level; few full-paper reads). Marked **[UNVERIFIED]** = from blog/README/secondary summary or no numbers seen. No Vietnamese-legal-specific paper with query-rewrite ablation numbers was found.

Our baseline facts: gold Điều in top-5 ~90%, top-20 ~97%; wrong cite -> rougeL ~0.40 vs right ~0.70. So retrieval headroom is ~10 pts at top-5 (~7 pts to top-20). Rewrites must beat a strong dense+reranker baseline; most literature gains are vs. weak BM25.

## Key takeaway
- Literature gains are largest for BM25/weak retrievers; dense + cross-encoder reranker leaves little. Expect +0-3 pts top-5 from generic rewriting; risk of drift is real.
- Highest-leverage query-side ideas for us are NOT generative rewrites: they are (1) extract named law/number/Điều from question -> boost/filter, (2) fuse original + rewrite candidates (never replace), (3) use the question profile for prompt conditioning, (4) oracle-ish "retrieve more, rerank better" (top-20 recall 97% vs top-5 90%).

## (a) Query rewriting / expansion

### HyDE / Query2doc (pseudo-document)
- What: LLM writes hypothetical answer/passage; embed/append it to query.
- Gains: Query2doc +3-15% BM25 on MS MARCO/TREC DL; dense DPR-base ~+1-4 nDCG@10 on TREC DL19; zero-shot BEIR mixed (gain DBpedia, drops NFCorpus, SciFact). https://ar5iv.labs.arxiv.org/html/2303.07678
- Small models: DEPT (3B generator, 5 BEIR tasks) avg nDCG@10 Query2doc ~37.7, HyDE ~33.8, original ~36.2 => HyDE hurts with 3B. https://arxiv.org/pdf/2608.17632 (single paper)
- Cost: 1 LLM call (generation 100-300 tokens), ~0.2-0.5 s on small model [UNVERIFIED blog].
- Risks: hallucinated article numbers/amounts; anchors in wrong law. Legal domain knowledge in Vietnamese in a <4B model is weak.
- Fit: Prompt model to write "Căn cứ tại khoản N Điều M Thông tư X quy định ..." style pseudo-answer WITHOUT numbers; embed concat(q, pseudo); fuse with original-q ranking via RRF. Teacher (Haiku) generating pseudo-answers offline for dev set = upper bound test for small-model viability.

### LLM query expansion best practices
- Jagerman et al.: LLM expansions beat classic QE on MS MARCO/BEIR; CoT prompts help; gains shrink for strong retrievers. https://ar5iv.labs.arxiv.org/html/2305.03653 ; Best practices paper https://arxiv.org/html/2401.06311v3
- Survey notes CoT expansion remains effective with smaller models. https://arxiv.org/pdf/2509.07794
- MuGI: training-free expansion lifts 23M retriever past 7B baseline [from search summary; strong-retriever case not shown].

### Multi-query / RAG-Fusion / DMQR-RAG
- Multi-query generally > single rewrite; DMQR-RAG (diverse rewrite strategies, info-based selection) beats RAG-Fusion. https://arxiv.org/html/2411.13154v1
- Blog claims of 8-10% [UNVERIFIED].
- Cost: 1 call (N rewrites) + N retrievals (cheap: 724k-chunk ANN) + RRF; reranker call count grows if reranking union (rerank union top-50 once).
- Fit: keep original q as one list; add 2-3 rewrites (legal-term paraphrase, keyword-only, "Điều ... Luật ..." style); RRF; rerank union. Safest rewriting form since original ranking is preserved.

### Step-back / decomposition
- Step-back (abstract the question) helps multi-hop/reasoning; our questions are single-article lookups (reference answer quotes one Điều) so low expected value. Decomposition only for multi-law questions (unknown share; measure). Evidence in this research: none legal-specific.

### Keyword extraction rewriting
- Strips long NL question (median 137 chars) to key legal terms; helps lexical retrieval, little for bge-m3 which handles NL well. Use as additional list in fusion (BM25/bge-m3 sparse). Cheap with small model or rule-based (regex + noun phrase).

### Legal-domain rewriting / implicit concept inference
- COLIEE 2022/2023 statute retrieval: LLM-inferred legal terms/facts, term expansion + reformulation, lexical+semantic ensemble beat all participating teams (abstract-level; no numbers seen). https://arxiv.org/abs/2410.12154
- Vietnamese: arXiv 2409.13699 Vietnamese legal QA: normalized RRF of keyword+vector, Active Retrieval reranking; abstract gives no numbers. https://arxiv.org/abs/2409.13699 (an LLM-refined question step is described in snippet only)
- HF README hue-portal uses Qwen2.5-1.5B rewriting + bge-m3; accuracy claims [UNVERIFIED]. 

### Hybrid caution
- Cross-lingual Vietnamese labour law study: hybrid RRF (BM25+dense) underperformed dense on all metrics; bge-m3 dense slightly above its hybrid on a Vietnamese legal benchmark model card [UNVERIFIED metric]. https://arxiv.org/pdf/2609.27376 => lexical side-lists must be weighted lightly, test on dev.
- VLSP 2025 DRiLL systems: BM25+bge-m3 hybrid, top-50 -> bge-reranker-v2-m3. https://preview.aclanthology.org/anlp-naacl/2025.vlsp-1.19.pdf (our pipeline is the same shape).

### RL-trained rewriters
- DeepRetrieval (3B): recall 65.07% vs prior SOTA 24.68% (publication search, PubMed), 63.18% vs 32.11% (trial search); reward = retrieval metric; v1 numbers differed (60.82/70.84). Gains are on boolean-query search engines, not dense retrieval. https://arxiv.org/abs/2503.00223
- Counter-evidence: controlled study found vanilla on-policy RL query rewriting gives limited gains, often beaten by simple prompting; proposes OPQE (RL-trained pseudo-document generation) which beats both. https://arxiv.org/abs/2510.17139 (no numbers seen)
- ConvSearch-R1 (3B, GRPO, rank-incentive reward RIRS): >10% gains on TopiOCQA/QReCC, conversational rewriting only; plain MRR/NDCG reward is sparse -> graded rank reward needed. https://arxiv.org/abs/2505.15776
- RAG-Gym: not found by search; unverified.
- Fit for us: reward = rank of gold Điều in dense top-50 (graded) + reranked top-5 hit. We have gold labels for train questions, so reward is free and cheap (no LLM judge). Rank-shaped reward is directly applicable. But headroom (10 pts) and baseline-strength argue for doing this AFTER prompt-only rewriting; note memory: GRPO is Tier 2.
- Cost: training only; inference 1 short call.

## (b) Question profiling (hint block)
- Evidence: Adaptive-RAG trains a small LM classifier on auto-labels to route no-retrieval / single / multi-step by complexity. https://alphaxiv.org/overview/2403.14403v2 (secondary summaries; no numbers seen) [UNVERIFIED gains]. 2026 complexity classifier 85.9% acc [UNVERIFIED, different system].
- For us, tiers collapse: nearly all questions are single-article lookups; the useful profile fields are: named doc (loại + số/ký hiệu + năm), legal domain, answer shape (mức/tỷ lệ, thời hạn, điều kiện, thủ tục), validity/amendment cue ("hiện hành", "sửa đổi"), answerability.
- Plug-in: (1) regex/NER fields -> retrieval boost (see c); (2) answer-shape -> choose shots (shots matching shape, already in C1 pipeline); (3) amendment cue -> include newer docs/ date tie-break; (4) dense-vs-reranker score margin as *retrieval-confidence* feature -> choose k (5 vs 10) or abstain; this needs no LLM.
- Risk: small-model profile errors propagate into prompt; mitigate by computing fields with regex first, LLM only for fuzzy ones; ablate hint block (does the SFT model even use it? needs SFT with hints in-train).
- Cost: regex = 0; LLM profile = 1 short call (or done in same SFT pass as a prefix).

## (c) Law number / entity extraction -> filtered or boosted retrieval
- Rationale: questions often name the document ("Thông tư X", "Nghị định số ...") -> doc-level constraint is far more precise than semantics. Wrong-Điều failures are often right-law-wrong-article or wrong-law-same-topic (to verify on our dev errors).
- Evidence: Stanford RegLab "Hallucination-Free?" shows legal RAG tools still hallucinate (>=1 in 6 queries; exact % [verify]); finding the right authority is the hard part. https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries
- Metadata-enriched legal RAG (UMass, arXiv 2603.19251) + DPO refusal: https://arxiv.org/html/2603.19251 (numbers not seen).
- Practitioner: BM25/lexical for citation tokens ("42 U.S.C. 1983") because dense misses them [UNVERIFIED blog].
- Design: parse doc type + number/year + issuer from question; match to corpus metadata (title/so_hieu); (i) hard filter only when exact-match confidence is high, fallback to unfiltered; (ii) otherwise additive score boost before rerank (candidate Điều from matched doc enter top-50 pool). Also explicit "Điều M" / "khoản N" mentions -> inject Điều M of matched doc.
- Cost: ~0 (regex + metadata index). No LLM. Risk: abbreviations/aliases (TT-NHNN, "Luật Các TCTD 2024"), amended docs sharing title, wrong filter excluding gold (use boost, measure filter recall).
- Measure first: fraction of dev questions naming a doc; recall@5 on that subset; oracle upper bound (we already have an oracle eval mode: gold Điều at rank 1).

## (d) Vietnamese legal-specific evidence
- Zalo Legal 2021 (~2.5k train/640 eval) used as benchmark in https://preview.aclanthology.org/moar-dois/2025.naacl-short.12.pdf ; VNLAWQC synthetic queries via Llama-3-70B.
- BKAI synthetic legal queries for bge-style training: https://arxiv.org/abs/2412.00657
- ALQAC 2024 (retrieval + QA, F2 macro); UIT-DarkCow 3rd retrieval/2nd QA (no QE details seen) https://sites.google.com/view/ALQAC-2024 ; ALQAC 2025 restricts external QA-annotated data.
- VLSP 2025 DRiLL: Qwen2.5-72B summaries for law/article-level then clause retrieval (hierarchical idea) https://preview.aclanthology.org/anlp-naacl/2025.vlsp-1.19.pdf
- Not found: a Vietnamese paper with clean ablation of HyDE/multi-query on Zalo/ALQAC; VN-MTEB legal query-rewrite results not found. Gap => our own ablation is itself novel.

## Ranked shortlist (cheapest first test)

1. Doc-name/number/Điều extraction -> boost/inject (c). EV: high, cost ~0. Test: regex on dev questions; report coverage, recall@5 on covered subset with boost vs without; check filter-vs-boost. Expect +1-3 pts top-5 overall (bigger on covered subset) [estimate].
2. Multi-query fusion (original + 2-3 small-model/teacher rewrites, RRF, rerank union) (a). EV: medium (+0.5-2 pts top-5; more at top-20). Test: Haiku-generated rewrites for dev set (upper bound); if <0.5 pt, drop; if positive, repeat with Qwen3.5-2B base/ SFT.
3. HyDE-style "answer skeleton" pseudo-doc in Điều-quote style, fused with original (a). EV: medium-low with <4B (DEPT: HyDE hurt at 3B). Test: teacher pseudo-answers (no numbers) concatenated; compare recall@5/20; then 2B generator.
4. Confidence-based adaptive k / profile hint (b): dense-reranker margin picks k=5 vs 8-10, or flags low-confidence for abstain/ multi-cite. EV: medium on rougeL since wrong cite costs ~0.30. Test: bin dev questions by reranker top1-top2 margin; plot gold-in-top-k and rougeL by bin; offline simulate best k policy (no new LLM calls).
5. RL rewriter w/ rank-shaped reward (DeepRetrieval/ConvSearch-R1 style) (a). EV: uncertain/low-med given 10-pt headroom and counter-evidence (2510.17139). Only after 2/3 show positive prompt-only gain; reward = graded rank of gold Điều in fused list; Tier 2 per project plan. Cheapest test: best-of-N rewrite oracle (sample 8 rewrites per train q, measure how often some rewrite lifts gold into top-5) = ceiling for RL.
