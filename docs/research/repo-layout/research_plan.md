# Research: reorganising a Python ML/research repo with folders that grew too large
Question: how should a RAG + small-LLM fine-tuning research repo (src/{corpus,index,retrieve,generate,train,evaluate,kg,ops}, ~45 scripts, ~110 queue files, docs/research, research_* folders at the root) be organised, and how to migrate safely while queues on a remote box reference script paths?
Subtopics: (1) conventions and templates (Cookiecutter Data Science, Lightning-Hydra, src layout, experiment/queue file management, archive); (2) safe migration (git mv, import compatibility, path references in configs/queues, running processes, index README files).
Synthesis: proposed target tree for this repo + migration order + what to leave alone.
