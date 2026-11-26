# Research plan: an effective CPT recipe so a 2B model really learns a small law corpus

**Question.** Which concrete CPT recipe (data design, objective/loss, optimisation, evaluation) makes a ~2B model memorise AND be able to use the facts of a small corpus (7-50M tokens of Vietnamese law articles) within a few hours on 2 x L40S, as a step before our normal RAG-SFT?

## Subtopics
1. **Data design**: paraphrase/rewriting, EntiGraph / synthetic CPT, per-article QA, knowledge augmentation, repeat counts per fact, format with law name + article number, ordering/curriculum, results on 1-8B models with few-M-token corpora.
2. **Objective and loss**: full-text next-token vs answer-only loss, instruction pre-training / instruction modelling, mixing instruction data with raw text, reversal curse and bidirectional/reverse augmentation, cloze/completion tasks, retrieval-augmented CPT (context in the input during learning).
3. **Optimisation + measurement**: full vs LoRA/PiSSA/rsLoRA for knowledge, lr/warmup/schedule, replay ratios, WiSE-FT / weight averaging / model merging after CPT, early stopping by recall, closed-book probes, cloze on articles, held-out paraphrase questions, with vs without RAG.

## Synthesis
A ranked, feasible recipe for 2 L40S in a few hours, and how to measure it on the gold-missing group.
