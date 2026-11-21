# Legal RAG verification: findings (web search, 3 searches)

Caveat: facts come from search-result summaries, NOT full-paper reads. Numbers unverified; re-read before citing in report.

## Key facts

### Vietnamese legal (closest match)
- NLLP 2026, "Cross-Lingual Legal QA for Vietnamese Labour Law: Retrieval, Translation, and Verifier-Guided Correction". 231 VI-EN QA pairs, Vietnamese labour law. Pipeline: split answer into claims -> check cited statute reachable + entails claim -> fix citation failures.
  - Result: verifier-guided correction improves citation preservation by +0.022 to +0.034 (system level); "no reliable gains" on other dimensions.
  - => faithfulness/citation gain only, no answer-quality gain.
  - https://aclanthology.org/2026.nllp-1.21/ , https://arxiv.org/pdf/2609.27376
- VLQA: >3,000 VI legal questions, ~59k provisions corpus, legal-professional annotated. https://arxiv.org/html/2507.19995v1
- Vietnamese legal IR for QA (RAG vs hallucination; many articles exceed 256-token embedder limit). https://arxiv.org/abs/2409.13699
- No off-the-shelf Vietnamese citation checker found.

### Generator + checker pipelines (general/legal)
- VeriCite: generator proposes evidence, NLI model verifies before attaching citations. Not legal. https://arxiv.org/html/2510.11394v1
- CiteGuard-RAG: sentence-level check that citation points to retrieved block + lexical/semantic similarity threshold; regenerate if fail. Ablation: strong retrieval alone does not give valid citations. https://www.alphaxiv.org/abs/2609.15830
- Adaptive legal RAG (UCL): reports RAGAS faithfulness/relevancy + BERTScore + ROUGE-Recall; pipeline tuning, not verifier. https://arxiv.org/pdf/2508.13107
- ACL 2025 legal reasoning with process verifier (step-level check). https://preview.aclanthology.org/landing_page/2025.acl-long.361.pdf
- LawThinker: self-reflection shares the context that produced the error -> use separate verifier. https://arxiv.org/pdf/2602.12056

### Quality (ROUGE) vs faithfulness
- No source found reporting ROUGE gain from a legal citation verifier. Reported gains are in citation validity/faithfulness.
- ROUGE/BLEU called inadequate for legal text: https://arxiv.org/html/2601.15182v1
- IJCAI 2026 (UIUC): semantic metrics alone insufficient for legal RAG; explicit article citation is strongest quality predictor. https://www.ijcai.org/proceedings/2026/0659.pdf

### Small (<4B) verifiers / judges
- SEVA: 3B fact-attribution verifier, structured evidence alignment, claimed ~GPT-4o-mini level on ClearFacts (69.0). Not legal. Its iterative 7B refine loop: +15pp HaluEval but -10 to -14pp TruthfulQA (specialization, not monotone gain). https://arxiv.org/abs/2606.29713
- Compact judges (NeurIPS 2026): 4-bit AWQ cuts balanced accuracy of 7B legal evaluator by 0.026; <1B has ranking problems calibration does not fix. https://neurips.cc/virtual/2026/160516
- VerityGate: LLM judge prefers fluent ungrounded narration over verifier output. https://arxiv.org/pdf/2610.00833
- CrossSource (MIT harness, citation P/R): same-family judge/generator self-preference bias. https://github.com/zoeb-nomi/crosssource

## Implication for law-rag
- Expect verifier loop to raise citation/faithfulness, not rougeL. Pre-state this (see train-only-if-meaningful memory).
- Use separate, claim-level verifier prompt; avoid same-model self-judge; check iterative-loop degradation (matches dpo round-3 drop).
