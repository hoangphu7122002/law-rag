# Research plan: clean codebase for law-rag (2026-11-17)

**Question:** What layout lets a reader understand a RAG research project at a glance, while keeping experiments
reproducible and GPU backends swappable (Mac / Modal / L40S)? Reuse what works in datamate, drop what is messy.

## Subtopics
1. **Reference RAG research repos** — FlashRAG, RAGFlow, Haystack, LlamaIndex templates, AutoRAG, RAGLab,
   cookiecutter-data-science: folder layout, pipeline-as-components, config style.
2. **Experiment infra patterns** — Hydra vs pydantic-settings/YAML, run dir + ledger vs MLflow/W&B,
   reproducible eval harness (fixed split, seeds, cached retrieval), vLLM local vs remote (OpenAI-compatible endpoint).
3. **Datamate audit** (local, on m5 via ssh) — what layout/patterns datamate uses, what is clean, what is messy.

## Synthesis
Proposed law-rag layout + conventions in research_report.md, each choice tied to a source or a datamate lesson.
