# Findings: Vietnamese legal QA / RAG systems with generation

Collected by a research subagent (~6 searches, mostly snippet level). UNVERIFIED = from a search summary, not read at source.

## 1. AITeamVN Vi-Qwen2-{1.5B,3B,7B}-RAG (general Vietnamese RAG generator, not legal)
- https://huggingface.co/AITeamVN/Vi-Qwen2-3B-RAG (also 1.5B, 7B). 7B card: base Qwen2-7B-Instruct; 1.5B/3B base UNVERIFIED. Apache-2.0, ctx 8192.
- Recipe not published (no data size, LoRA/full, DPO/GRPO). Cite as ViRAG-Gen (2024), Nguyen Nho Trung, Nguyen Nhat Quang.
- Trained for noise robustness, negative rejection, multi-document integration; "Có/Không" relevance mode (~99 % claimed, UNVERIFIED).
- Prompt: Vietnamese system prompt + "answer only from context else refuse" + `### Ngữ cảnh` (numbered) / `### Câu hỏi` / `### Trả lời`; example temp 0.1, max_new_tokens 2048.
- Eval: human 10-point on own EvalRAGData; VMLU (7B) avg 56.04. No legal or citation metric.

## 2. Hybrid legal RAG paper — arXiv 2409.13699 (Sep 2024)
- Section split with overlap → hybrid keyword + vector with enhanced RRF → "Active Retrieval" rerank → LLM. Numbers UNVERIFIED.

## 3. Graph-based Vietnamese legal systems (numbers UNVERIFIED)
- Three-Tier KG RAG (ontology + KG + document-tree forest; LLM extraction refined by experts): https://journals.sagepub.com/doi/10.3233/FAIA260735
- Self-reflection Graph-RAG (generate → evaluate → refine, bounded): https://journal.futuristech.co.id/index.php/sjcs/article/download/380/64/910
- Vietnamese Legal GraphRAG (agentic, RRF, bge-reranker-v2-m3): https://ciel.is-a.dev/project/viet-legal-graph-rag/vietlegalgraphrag
- VLSP 2025 MLQA-TSR traffic law heterogeneous KG: https://aclanthology.org/2025.vlsp-1.49.pdf
- ViHERMES multihop healthcare-regulation QA, compares GraphRAG / HippoRAG / LightRAG / MiniRAG: https://arxiv.org/html/2602.07361
- Repos: https://github.com/tontide1/Legal-RAG (LangChain, BM25, Neo4j, NER, Gemini, vietnamese-sbert);
  https://github.com/yammdd/GraphRAG-Chatbot (hierarchical Chương→Điều chunking, graph DB). No LlamaIndex Vietnamese legal repo found.

## 4. Datasets / shared tasks
- VLQA (arXiv 2507.19995): 3,129 questions, expert answers with citations, ~59k provisions. https://arxiv.org/html/2507.19995v1
- ViLegalLM https://github.com/ntphuc149/ViLegalLM: hard negatives from ALQAC + Zalo, QLoRA decoders (mapping to 2507.09638 UNVERIFIED).
- VLSP 2025 LegalSLM: small legal LMs via CPT + SFT; Bosch@AI teacher → synthetic instruction / CoT → student PEFT. https://preview.aclanthology.org/vol-to-evt/2025.vlsp-1.22.pdf

## 5. Small-LLM RL for legal citations (Thai, nearest template)
- GRPO on NitiBench, reward = citation accuracy + response quality; up to 90 % citation-F1 gain; BGE-M3 similarity reward ~2.5× cheaper than LLM judge. UNVERIFIED, URL not captured.
- No Vietnamese DPO/GRPO-with-citation paper on < 4B found.

## Gaps
- No open Vietnamese system documents validity / effective-date filters.
- Query rewriting only implicit inside agentic loops.
