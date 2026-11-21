# Research plan: LLM checker (verifier) with feedback loop for RAG QA

Question: if a checker LLM verifies the answer and sends feedback to retriever / reranker / generator (duo / critic-refine loop), does ROUGE-L-style quality rise, esp. for <4B generators on template-style legal QA?

Subtopics
1. Retrieval-side feedback: Self-RAG, CRAG, adaptive/iterative retrieval, retrieval-evaluator results; gains and small-model viability.
2. Generator-side self-correction / critic-refine: limits of self-correction (Huang et al. 2023 and follow-ups), verifier/critic with small models, reward-model reranking (best-of-n with verifier), effect on lexical metrics.
3. Evidence for our setting: small-model (<4B) verifiers, legal RAG verification, citation verification, and our own measured oracle ceilings (best-of-8 oracle, gold-passage oracle +.004, cascade).

Synthesis: compare with our measurements (retrieval oracle +.004 rougeL, MBR/best-of-n oracle ceiling, DPO greedy .6637) and recommend whether to implement.
