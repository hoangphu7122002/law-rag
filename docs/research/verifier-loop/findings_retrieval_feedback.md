# Retrieval-side feedback in RAG: findings

Date: 2026-11-21. 3 searches + 2 page fetches. Numbers from Self-RAG/CRAG are fetched from arXiv HTML; others are search-snippet level (unverified).

## Self-RAG (arXiv 2310.11511, https://arxiv.org/abs/2310.11511)
- Mechanism: LM (7B/13B Llama2 fine-tuned) emits reflection tokens: Retrieve, IsRel, IsSup, IsUse; critic-distilled training; inference-time segment beam search w/ weighted critique scores (IsRel 1.0, IsSup 1.0, IsUse 0.5), retrieval threshold 0.2.
- Table 2 (acc unless noted), Self-RAG 7B / 13B vs baselines:
  - PopQA 54.9 / 55.8 (Llama2-7B retrieval 38.2; Ret-Llama2-chat-13B 51.8; Ret-ChatGPT 50.8)
  - PubHealth 72.4 / 74.5 (Llama2-7B retrieval 30.0)
  - ARC 67.3 / 73.1; Bio FactScore 81.2 / 80.2 (Llama2-7B retrieval 78.0)
  - ASQA str-em 30.0 / 31.7 (Ret-ChatGPT 40.7); **ASQA ROUGE 35.7 / 37.0** (Llama2-FT-7B 35.8, Alpaca-13B retrieval 36.7, Ret-ChatGPT 39.9); citation prec 66.9/70.3, rec 67.8/71.3 (Llama2 retrieval baselines: prec 2-3, rec 4).
- Reading: big gains are on accuracy and citation quality vs naive retrieve-then-read baselines that were NOT fine-tuned. On ROUGE (only ASQA) Self-RAG 7B (35.7) is ~equal to plain fine-tuned Llama2-FT-7B (35.8) -> reflection adds ~0 ROUGE.
- Ablation (project site https://selfrag.github.io): no retrieval -> PopQA ~40% relative drop, PubHealth only ~2%.
- Smallest model tested is 7B. No <4B results.
- RetrievalQA (https://arxiv.org/pdf/2402.16457), snippet: "effectiveness of Self-RAG largely depends on threshold tuning".

## Corrective RAG / CRAG (arXiv 2401.15884, https://arxiv.org/abs/2401.15884)
- Mechanism: lightweight T5-large evaluator (0.77B) scores each retrieved doc -> Correct / Incorrect / Ambiguous. Correct: knowledge refinement (decompose-filter-recompose strips). Incorrect: discard, web search. Ambiguous: both. Plug-and-play, no generator retraining.
- Evaluator: 84.3% retrieval-assessment acc on PopQA vs ChatGPT 58.0 / CoT 62.4 / few-shot 64.7.
- Table 1 (acc; Bio = FactScore):
  - LLaMA2-hf-7b: RAG -> CRAG: PopQA 50.5->54.9, Bio 44.9->47.7, PubHealth 48.9->59.5, ARC 43.4->53.7
  - SelfRAG-LLaMA2-7b: RAG 52.8/59.2/39.0/53.2 -> CRAG 59.8/74.1/75.6/68.6; Self-RAG 54.9/81.2/72.4/67.3 -> Self-CRAG 61.8/86.2/74.8/67.2 (ARC flat/slightly lower)
  - Gains: +2.8 to +14.9 pts over RAG; vs Self-RAG +0 to +7 pts (PopQA +6.9, Bio +5.0, PubHealth +2.4, ARC -0.1).
- Ablation PopQA (LLaMA2-hf-7b): full 54.9; w/o Correct 53.2; w/o Incorrect 54.4; w/o Ambiguous 54.0; only Incorrect (web) 47.0. Each action adds ~0.5-2 pts; web search alone hurts.
- Caveat quote: LLaMA2-hf-7b often failed to output valid True/False on PubHealth (Self-RAG* 0.7) -> instruction-following of the generator matters.
- Metrics: accuracy/FactScore only. No ROUGE.
- Independent reproduction, Phi-3-mini generator (3.8B), https://arxiv.org/pdf/2603.16169: PopQA 51.4 (RAG) -> 54.4 (CRAG) = +3.0; ARC-C 84.8 -> 85.2 = +0.4. Evaluator labels 88.3% of ARC questions "Ambiguous" (trained on biographical entity Qs) -> evaluator domain shift limits benefit. (Snippet-level, not read in full.)

## Adaptive / iterative / small-model retrieval gating
- Tiny-Critic RAG (arXiv 2603.00846, https://arxiv.org/html/2603.00846v1): Qwen3-1.7B + LoRA as retrieval gatekeeper. Routing F1 0.912 vs GPT-4o-mini "Heavy-CRAG" 0.934; TTFT 492 ms vs 1235 ms. Metrics: routing F1, RAGAS faithfulness. No ROUGE. Shows <4B can be the evaluator/gate, not necessarily generator. Preprint.
- ReflectiveRAG (EACL 2026 industry, https://aclanthology.org/2026.eacl-industry.27.pdf): small LM as iterative sufficiency controller, no policy training. Latency-focused. Snippet-level only.
- Self-Routing RAG (arXiv 2504.01018, https://arxiv.org/html/2504.01018v3): selective retrieval binding; not read.
- "RAG in the Wild" (ACL Findings 2026, https://aclanthology.org/2026.findings-acl.849.pdf), snippet: in mixed-knowledge settings retrieval benefit is largely confined to smaller LMs.

## Answers to the questions
1. Quantitative gains: real but on accuracy/FactScore/citation metrics, magnitude 3-15 pts over plain RAG with 7B models (CRAG, Self-RAG); CRAG over Self-RAG a further 0-7 pts.
2. Small generators (<4B): no direct Self-RAG/CRAG result found. Closest: Phi-3-mini (3.8B) CRAG repro +3.0 PopQA, +0.4 ARC. Self-RAG needs a fine-tuned 7B+ critic-generator; not designed for tiny models. Small LMs work better as the evaluator/gate (T5-large 0.77B, Qwen3-1.7B).
3. ROUGE-L / lexical overlap: effectively no evidence of gains. Only lexical metric is ASQA ROUGE in Self-RAG, where Self-RAG 7B ~= fine-tuned baseline (35.7 vs 35.8). Both papers' gains are on factual-correctness metrics; ROUGE-L rewards answer style/format, which retrieval filtering does not change much.
4. Implication for law-rag: retrieval-side feedback (CRAG-style evaluator + re-retrieval/filter) plausibly raises recall/faithfulness, but expect small rougeL movement unless evidence quality is the bottleneck. Worth an explicit pre-stated significance test (consistent with train-only-if-meaningful rule). Evaluator domain shift (ARC 88% Ambiguous) means a legal-domain evaluator needs in-domain training/calibration.
