# CPT strategy for Qwen3.5-2B on the laws behind our questions (synthesis of 3 subagent reports; most numbers from abstracts/snippets, not fully verified)

## Evidence
- Ovadia et al. 2312.05934 (7B models): raw-text fine-tuning barely helps and can hurt (Llama2 .353 -> .219 on current events; Mistral .481 -> .504); RAG is far better (.875); FT+RAG does not beat RAG on new knowledge; ~10 paraphrases per chunk help monotonically.
- EntiGraph 2409.07431 (Llama3 8B, 1.3M-token corpus): raw-text CPT scored BELOW the base model; synthetic corpus 455M tokens (350x) lifted QA 39.5 -> 56.2; with RAG 62.6 vs base+RAG 60.4.
- Physics of LMs 3.1 (2309.14316): without paraphrase/augmentation the model memorises but cannot extract facts. Capacity (2 bits/param) is not the limit at 3M tokens; exposure diversity is.
- Biderman 2405.09673: in CPT, LoRA learns less (and forgets less) than full FT; full FT lr ~1e-5..5e-5, LoRA ~10x that.
- Gekhman 2405.05904 (PaLM-2 S): fitting facts the model does not know raises hallucination; early stopping makes them neutral; relabel unknowns as "I don't know".
- Mecklenburg 2404.00213: cover every fact evenly with varied QA, not repeated text.
- Legal CPT (SaulLM 30B tokens, ViLegalLM 16 GB, LawInstruct T5): gains come at billions of tokens, mostly classification/MCQ; DACP Taiwan: not uniformly better; no source shows a measurable effect for a few-million-token CPT corpus after SFT, nor on ROUGE generation.

## Verdict for us
Raw-text CPT on ~3M tokens is unlikely to help and may hurt; knowledge injection with RAG already present is weaker than RAG. The target group (gold missing, ~10% of questions, 25 dev + 18 test) is too small to measure a gain below ~0.05 on the group (= 0.005 overall, under the 0.015 seed noise). If done anyway: full FT lr 1e-5..2e-5, paraphrases / QA per article (not raw repeats), 5-10% replay of our SFT rows, early stop, then SFT (PiSSA on C) from the CPT model; judge on hit and miss groups separately.
Priority: DPO from the best SFT first; CPT only as an optional probe if time remains.
