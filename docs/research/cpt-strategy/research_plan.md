# Research plan: continued pretraining (CPT) strategy for Qwen3.5-2B on Vietnamese law text, then SFT for RAG QA

**Question.** We want the answer model (Qwen3.5-2B) to "know" the laws our questions are about (about 570 documents, ~3M tokens, from the same corpus the retriever reads) so that the ~10% of questions where retrieval misses the gold article (ROUGE-L ~0.32-0.43 vs ~0.65 when gold is present; ceiling +0.03) get better. 2 x L40S, 3 days left. Which CPT strategy is worth it, if any?

## Subtopics
1. **How to inject knowledge by training**: LoRA vs full, lr, epochs on a small corpus (~3M tokens, many epochs), memorisation vs overfitting, replay/mixing, paraphrase / synthetic rewriting (Synthetic Continued Pretraining / EntiGraph, knowledge augmentation), data formats (article-level text vs per-article Q/A).
2. **Evidence: fine-tuning vs RAG for knowledge, hallucination risk**: Ovadia et al., Mecklenburg et al. (Injecting New Knowledge), Gekhman et al. (new knowledge in SFT encourages hallucination), "physics of LMs" knowledge capacity.
3. **Legal domain CPT/DAPT results and how it combines with RAG on small models**: SaulLM, LawLLM / lawyer-llama, ViLegalLM, legal-BERT, gain after SFT, CPT -> SFT order, starting from a base vs an SFT'd model.

## Synthesis
Recommendation for our box and deadline: do / do not, concrete hyper-parameters, what to measure (gold-missing group), risks.
