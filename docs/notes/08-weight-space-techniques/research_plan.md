# Research plan — techniques beyond distillation (2026-11-18)

**Question:** which weight-space / inference-time / knowledge-injection techniques could improve a Vietnamese law
RAG whose answering LLM must be < 4B (Vi-Qwen2-3B-RAG, Qwen3.5-2B), scored by cosine / Jaccard / token overlap /
BLEU-1 / ROUGE-L against reference answers that quote law articles ("Căn cứ khoản X Điều Y … quy định như sau: …")?
Trigger: "Sparks of Weight-Space Programming" (https://rikiyat.github.io/blog/weight-space-programming/).

| Subtopic | Expected | File |
|---|---|---|
| A. Training-free edits at inference / in weights: activation steering, representation engineering, weight orthogonalization, context-aware / contrastive decoding (CAD, DoLa), copy bias | which raise RAG faithfulness or verbatim quoting; cost; evidence on small models | findings_inference_edits.md |
| B. Putting knowledge into weights: hypernetwork / doc-to-LoRA (Text-to-LoRA, Drag-and-Drop, Generative Adapter), Cartridges / self-study, SEAL, synthetic continued pretraining (EntiGraph, Active Reading), knowledge editing (ROME/MEMIT/AlphaEdit), test-time training | does it beat RAG for domain QA; scale limits | findings_knowledge_injection.md |
| C. Model merging and weight averaging: task arithmetic, TIES, DARE, model soups, WiSE-FT, LoRA soups / LoraHub, merging an RAG model with a domain SFT model | gains for small domain models; tools (mergekit) | findings_merging.md |

Synthesis: research_report.md — ranked list for law-rag (gain × cost × risk), mapped to roadmap ids.
