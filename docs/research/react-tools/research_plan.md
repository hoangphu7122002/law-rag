# Plan: tool design for small-LLM ReAct in legal RAG (user: "not more tools = better")
Question: how many / which tools, how to describe them, when do agents beat single-shot RAG for <4B models, legal evidence for get_article / graph / web.
Subtopics
1. small_model_tool_use: tool-count vs accuracy for 1-4B models, schema/description design, native function calling vs prompted ReAct (Qwen3/Qwen3.5), failure modes + mitigations (loops, malformed calls, context overflow), tool-set minimalism evidence (e.g. tool retrieval, ToolBench/BFCL small-model results).
2. agentic_retrieval_vs_single_shot: when iterative / adaptive retrieval (Self-RAG, query rewriting, multi-step, agentic RAG) helps vs hurts versus single-shot RAG; gains and costs with small models; multi-hop and legal QA benchmarks.
3. legal_tools: statute / article lookup, citation-graph or neighbor expansion, amendment/validity, web search for legal QA; reported gains (Vietnamese, ALQAC, COLIEE, LegalBench-RAG, others).
Synthesis: recommended minimal tool set + ablation order, risks, gaps.
