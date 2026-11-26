# Plan: ideas to raise ROUGE-L (2026-11-26)
Q: new (2025-26) techniques for a 2B Vietnamese legal QA generator whose answers quote provisions verbatim; bottleneck = quote extent + which-khoản; already tried SFT, on-policy DPO, CPT, MBR, soup, wide context.
1. Span-copy / extractive-citation generation and constrained decoding (quote-extent control).
2. Preference/RL with length/extent-aware or ROUGE-like rewards; RFT/ReST/best-of-n distillation; small-data tricks.
3. Retrieval/selection-side: clause-level selection/reranking, answer-span selection, steering/activation editing for style/extent.
Synthesis: ranked list vs. what we already tried, cost, expected gain.
