# Findings: training / inference class patterns (2026-11-17)

Source: research subagent. Sources: huggingface.co/docs/trl/sft_trainer · transformers callback/trainer docs ·
lightning.ai datamodule docs · transformers add_new_pipeline · eugeneyan.com/writing/design-patterns · /more-patterns

- **TRL SFT:** `SFTConfig` (extends TrainingArguments) + `SFTTrainer(model=id, args, train_dataset, peft_config=LoraConfig)`;
  prompt/completion or messages datasets; completion-only loss default for prompt/completion; LoRA lr ~1e-4; callbacks observe only.
  Subclass Trainer only if the loss changes.
- **Lightning:** separate model / data / loop — borrow the idea, not the classes (TRL owns the loop).
- **HF Pipeline:** Template Method preprocess → _forward → postprocess; vLLM `SamplingParams` value object.
- **Patterns:** Strategy (swap retriever/LLM), Factory dict (≥ 3 variants), Adapter (non-OpenAI APIs), Facade (`RAGPipeline.answer`),
  Protocol for interfaces / ABC for shared skeleton, frozen config objects. Composition over inheritance; avoid god classes,
  > 2-level inheritance, hidden globals, abstractions before a 2nd implementation.
- **Shared `Prompt`** used by both SFT data and inference so formats cannot drift; `Run` only owner of result paths.
