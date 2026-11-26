# src/train/

Training code, grouped by method (a flat folder of 26 modules before 2026-11-26; old -> new in [docs/MOVED.md](../../docs/MOVED.md)).

| package | purpose |
|---|---|
| [sft/](sft/README.md) | SFT data, context mixing, LoRA/PiSSA/rsLoRA/full SFT, teacher rows, checkpoint merging |
| [preference/](preference/README.md) | DPO pair building, DPO, GRPO |
| [cpt/](cpt/README.md) | continued pretraining (data, cleaning, augmentation, mix, probe, trainer) |
| [react/](react/README.md) | ReAct student data, repair, SFT, trajectory stats |
| [retrieval/](retrieval/README.md) | embedder and reranker fine-tuning |
| [synth/](synth/README.md) | synthetic Q/A generator core |
