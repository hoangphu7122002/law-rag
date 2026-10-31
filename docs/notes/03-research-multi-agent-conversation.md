# 03 — Research: multi-agent RAG with conversation and memory (2026-10-31)

Web research, 2 rounds, ~15 pages (summary only).

## Findings

- **Agent design:** Anthropic recommends simple workflows (routing, orchestrator-workers, evaluator-optimizer) over free
  agents; agents trade latency/cost for quality and compound errors (high).
- **Conversational RAG:** MTRAG (arXiv 2501.03468): 110 conversations, 842 tasks; systems struggle on later turns,
  non-standalone and unanswerable questions (high). LexRAG (arXiv 2502.20640): 1,013 five-turn legal dialogues,
  best Recall@10 33.33 % with query rewrite; concatenating full history hurts dense retrievers (high).
- **Memory layers:** Mem0 (2504.19413) +26 % over OpenAI memory on LoCoMo; Zep/Graphiti (2501.13956) temporal KG,
  LongMemEval up to +18.5 %; Letta: plain files + grep 74.0 % on LoCoMo. Vendor numbers conflict (medium/low).
- **Legal assistants:** Stanford (2405.20362): Lexis+ AI and Ask Practical Law AI wrong > 17 %, Westlaw AI > 34 %;
  failure types incorrect, misgrounded, inapplicable (overruled) law, sycophancy (high). Vietnamese agentic legal
  advisor (router + specialized agents, vector + graph + web): 82.3 % accuracy, no baseline (medium).
- **Small models as agents:** Qwen2.5-1.5B tool calls 75–79 % only with JSON recovery, 5/1,000 raw outputs strict JSON
  (arXiv 2609.07370, medium) → use constrained decoding and rule fallbacks.

## Recommended design

Fixed LangGraph workflow (not a free supervisor): router (rules/small classifier: chit-chat · follow-up · new legal
query · memory) → history-aware rewriter (last 2–3 turns + slot tracker law/article/date) → hybrid retrieval
(Qdrant + Neo4j, temporal filter) → generator → verifier (cited article exists, quote matches). Every role uses
vLLM guided JSON, retry once, then rule fallback. Memory: session summary + pinned slots + small per-user fact store;
Mem0/Graphiti only as an optional comparison. Evaluate with a small LexRAG-style multi-turn Vietnamese set (20–30 chains):
Recall@10 last-query vs rewritten, citation validity, abstain rate.

## Sources

https://www.anthropic.com/engineering/building-effective-agents · https://arxiv.org/abs/2501.03468 ·
https://arxiv.org/html/2502.20640v1 · https://arxiv.org/abs/2504.19413 · https://arxiv.org/abs/2501.13956 ·
https://www.letta.com/blog/benchmarking-ai-agent-memory · https://arxiv.org/abs/2405.20362 ·
https://www.scitepress.org/PublishedPapers/2025/137354 · https://arxiv.org/abs/2609.07370
