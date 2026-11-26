# src/train/sft/

supervised fine-tuning: data rendering, context mixing, LoRA/PiSSA SFT, teacher rows, checkpoint merging.

Import as `train.sft.<module>`; run as `python -m train.sft.<module>` or `python src/train/sft/<module>.py` with `PYTHONPATH=src`.

| module | what |
|---|---|
| `ctx_mix.py` | Context mixing for SFT rows (docs/research/tail-mid-plan.md): pure helpers on lists of passages. |
| `merge.py` | Training-free merging of full model checkpoints of one architecture (soup, task arithmetic). |
| `sft_data.py` | SFT / RAFT data: train questions (decontaminated) + their retrieved passages, rendered with the SAME prompt and chat |
| `sft_lora.py` | LoRA SFT on prompt / completion rows (src/train/sft/sft_data.py), loss on the completion only; then merge and save a |
| `teacher_rows.py` | Verified teacher answers as extra SFT rows (a second answer per train question). |
