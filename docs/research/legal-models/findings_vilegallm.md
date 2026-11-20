# ViLegalLM (ntphuc149) — primary sources (agent report 2026-11-20; HF cards, GitHub README, ACL PDF tables 4/5/8 read directly)
Paper: aclanthology.org/2026.findings-acl.1801 · github.com/ntphuc149/ViLegalLM · HF ntphuc149/ViLegalQwen3-1.7B-Base, ViLegalQwen2.5-1.5B-Base

| Model | Params | Base | Context | Objective |
|---|---|---|---|---|
| ViLegalBERT | 135M | PhoBERT-base-v2 | 256 | MLM |
| ViLegalQwen2.5-1.5B-Base | 1.54B | Qwen2.5-1.5B | 2048 | CLM |
| ViLegalQwen3-1.7B-Base | 1.72B | Qwen3-1.7B-Base | 4096 | CLM |
Base-only (not instruct); Apache-2.0, research use; 2.5 card gated. Corpus 16 GB (19 GB raw, 1.14M docs; thuvienphapluat, vbpl, luatvietnam, lawnet), released as ntphuc149/ViLegalText; training stopped early at 22k / 25.5k steps.

Protocol: per-task QLoRA instruction tuning (r16 a32); gpt-4o-mini is the only 0-shot row.
ViBidLQA-AQA (bidding-law abstractive QA, test 1000): ROUGE-L / BLEU-4 / BERTScore-F1
- Qwen2.5-1.5B 73.02 / 49.12 / 89.74 · ViLegalQwen2.5-1.5B 73.45 / 49.90 / 89.91
- Qwen3-1.7B-Base 73.81 / 51.18 / 90.24 · qwen3-1.7b-legal-pretrain (2.5 GB legal pretraining, other paper) 74.84 / 51.49 / 90.32 · ViLegalQwen3-1.7B 74.49 / 52.11 / 90.43
- gpt-4o-mini 0-shot 67.06 / 40.46 / 86.85
Legal pretraining gain over plain Qwen on generation: ROUGE-L +0.43 (2.5) and +0.68 (Qwen3), BERTScore < +0.2; no seeds/variance reported (likely noise). Classification gains larger: ALQAC-MCQ F1 85.03 -> 88.92, VLSP-NLI 95.64 -> 97.83.
Unverified: whether the AQA prompt includes the context passage, exact chat template, tokens seen; ALQAC-TF F1 83.34 (card) vs 82.35 (paper).
