# Research plan: class-based design patterns for law-rag (2026-11-17)

**Question:** Which class-based patterns make RAG + training + inference code easy to read for a small solo project?

## Subtopics
1. **RAG component classes** — LlamaIndex (BaseRetriever, postprocessors), Haystack 2 components, FlashRAG
   (BaseRetriever/BaseGenerator/BasicPipeline), DSPy Module/Signature: interface shape, how they compose.
2. **Training + inference classes** — HF Trainer (callbacks, TrainingArguments), Lightning LightningModule/DataModule,
   TRL SFTTrainer, HF pipeline (preprocess/_forward/postprocess Template Method), vLLM LLM class.
3. **Classic patterns applied to ML** — Strategy, Factory/Registry, Template Method, Adapter, Protocol vs ABC,
   dataclass configs; pitfalls (deep inheritance, god classes).

## Synthesis
Class map for law-rag (one base class per stage, concrete classes per method, Pipeline/Trainer/Evaluator) in research_report_oop.md.
