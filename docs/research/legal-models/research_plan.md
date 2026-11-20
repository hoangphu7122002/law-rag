# Research plan — Vietnamese legal LLMs under 4B: what do the ORIGINAL sources say? (2026-11-20)

Main question: for each candidate legal model (< 4B params), what do the primary sources (original Hugging Face model cards, the paper / GitHub repo, not mirror sites) say about: parameter count, base model, training data and method, whether instruct/chat or base only, chat template, license, and any reported benchmark (esp. generation metrics such as ROUGE-L / BLEU on legal QA)? Is any of them a better answer generator than our SFT Qwen3.5-2B (dev ROUGE-L ~.65)?

Subtopics (2 parallel agents, 3-5 searches each):
1. **ViLegalLM family (ntphuc149)** — paper (aclanthology.org/2026.findings-acl.1801), GitHub ntphuc149/ViLegalLM, HF cards of ViLegalQwen3-1.7B-Base and ViLegalQwen2.5-1.5B-Base: corpus (16GB), pretraining recipe, benchmarks and which tasks (generation/QA ROUGE?), license, chat template, comparison to base Qwen.
2. **Community legal models** — HF cards for bqbbao6/Qwen2.5-3B-legal-vn, thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2, VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain (+ its VLSP 2025 paper aclanthology 2025.vlsp-1.22), ntkhoi/Llama3.2-3B-Vietnamese-Law: data, method, eval, license, whether full weights.

Synthesis: docs/research/legal-models/research_report.md — table (model, params, base/instruct, data, benchmark, license, source URL, verified from primary source?) + verdict: which one (if any) is worth one SFT trial with our recipe. Claims from snippets/mirrors marked UNVERIFIED.
