# Self-correction / critic / verifier loops: findings

Source tags: [R] = retrieved by web search this session (search snippets, not full-paper reads); [M] = from model memory, UNVERIFIED, check before citing.

## 1. Intrinsic self-correction does not work (reasoning)
- Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" (ICLR 2024). https://arxiv.org/abs/2310.01798 [R]
  - "LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction."
  - GSM8K, GPT-3.5: 75.9% -> 74.7% after 2 rounds. Llama-2: 62.0% -> 36.5% (weaker model, worse drop).
  - GPT-3.5 flips correct->wrong 8.8% vs wrong->correct 7.6%.
  - With oracle correctness labels: 75.9% -> 84.3%. Bottleneck = verification, not generation.
  - [M] Also reports multi-agent debate ~ no better than self-consistency at equal cost; "improvements" in prior work came from weak initial prompts or oracle stopping.
- Counterpoint: "LLMs have Intrinsic Self-Correction Ability" https://arxiv.org/pdf/2406.15673 [R title only; gains need zero temperature + fair prompts, not read in detail].

## 2. Survey: when it works
- Kamoi et al., TACL 2024, "When Can LLMs Actually Correct Their Own Mistakes?" https://arxiv.org/abs/2406.01297 [R]
  - "no prior work shows reliable evidence of successful self-correction with in-context learning" for general tasks.
  - Works when reliable external feedback exists (code exec, tools, retrieval) [R]; fine-tuning works with large training data, small-data case unexplored [R].
  - Prior studies often use unfair evaluations (oracle labels, weak initial prompts) that over-credit self-correction [R].

## 3. Small models (<=13B)
- Zhang et al., "Small Language Models Need Strong Verifiers to Self-Correct Reasoning" (ACL Findings 2024). https://arxiv.org/abs/2404.17140 [R]
  - SCORE: small model generates its own critique/correction data, then fine-tunes; tested on LLaMA-2-13B-chat, Gemma-7B, 5 math+commonsense sets.
  - "notable performance gains when paired with a strong GPT-4-based verifier, though limitations are identified when using a weak self-verifier for determining when to correct."
  - Takeaway: small-model critic/verifier is the bottleneck; self-verification decides wrongly when to edit.
- Self-Refinement from external proxy-metric feedback, https://arxiv.org/pdf/2403.00827 [R]: works with Flan-T5-XXL and Llama-2-13B-Chat in content-grounded generation when feedback is an external proxy metric (not the model's own judgment).
- [M] Madaan et al. Self-Refine (2023): gains for GPT-3.5/4, reportedly little or negative for Vicuna-13B (fails to follow feedback format). Not verified this session. No retrieved evidence at <4B; extrapolation: weaker than the 13B case.

## 4. Verifier best-of-n reranking
- Gao et al., "Scaling Laws for Reward Model Overoptimization" https://arxiv.org/abs/2210.10760 [R]: proxy-RM score keeps rising under best-of-n while gold score peaks then falls; overoptimization occurs with both BoN and RL. Gain depends on RM size.
- Underspecification paper https://arxiv.org/pdf/2312.09244 [R]: reranking with reward models can amplify RM errors.
- 2026 preprint https://arxiv.org/pdf/2609.09776 [R, unreviewed]: learned verifier gold reward peaks ~N=512 then declines to 0.104; execution-based verifier rises monotonically to 0.831 at N=2048.
- Blog https://dev.to/ji_ai/best-of-n-sampling-why-n64-scores-higher-and-answers-worse-36gl [R, non-peer-reviewed]: best N for a given judge usually << 64; check with independent scorer.
- Rule: verifier gain is bounded by verifier quality and holds at small N; with a learned/weak verifier large N hurts.

## 5. Open-ended / long-form / lexical metrics
- Nothing retrieved directly addressing ROUGE/BLEU change after critique-refine. Gap.
- Stanford CS224R report (Qwen2.5 critique-guided refinement) [R]: refinement improved 12.5% of responses, avg reward +1.98; reward-model metric, not ROUGE. https://cs224r.stanford.edu/projects/pdfs/cs224r_final_report%20(16).pdf
- [M/inference, UNVERIFIED] LLM-judge/reward gains from refinement usually come from added detail, hedging, restructuring and paraphrase. Those move output away from a fixed reference, so rougeL tends flat or down while judge scores go up. Refinement also lengthens answers, hurting precision.
- Implication for law-rag: ROUGE-L is reference-bound; a critic loop must be checked on rougeL directly, not on judge/recall. Safe variants: (a) selection-only (BoN rerank by a metric-correlated scorer, no rewriting), (b) edit-gating (only revise when verifier confidently flags error), (c) external grounded feedback (citation/entailment check vs retrieved passages).

## Bottom line
1. Self-critique without external signal: ~0 to negative (-1 to -25 pts on GSM8K by model strength).
2. Gains need external/strong verifier or oracle (+8 pts with oracle label).
3. Small critic = weak verifier -> wrong edits; strong verifier needed.
4. BoN reranking: gain real at small N, degrades with large N and weak verifier.
5. Lexical metrics: no direct evidence found; expect rewrite to hurt rougeL.
