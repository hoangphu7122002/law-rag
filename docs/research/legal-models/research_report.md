# Vietnamese legal LLMs < 4B — worth a trial in law-rag? (2026-11-20)
Sources: findings_vilegallm.md (primary: paper tables + HF cards), findings_community.md (HF card summaries; Llama3.2-3B page 401 = unverified).

## Verdict
No. Legal continued pretraining helps classification (MCQ / NLI, +4 / +2 F1) but barely generation: on legal abstractive QA (ViBidLQA-AQA) ViLegalQwen adds +0.4 to +0.7 ROUGE-L over the plain Qwen of the same size after the same fine-tuning, no variance reported. Community models (bqbbao6 3B, thangvip GRPO 1.7B, LegalSML 1.7B pretrain) report no generation metric at all. Our answers copy the retrieved Dieu, so legal knowledge is supplied by retrieval; the gap to ~.73 best-of-8 is wording, not knowledge. Expected effect of swapping the base model: below the MBR noise (+-.006), at the cost of ~1 h SFT + eval per model.
If one trial is still wanted: ViLegalQwen3-1.7B-Base (most downloaded, ACL paper, Apache-2.0 research) with the v3b recipe; check its chat template first (base-only).
Side note: published fine-tuned 1.5-1.7B models reach ROUGE-L 73-75 on ViBidLQA-AQA, a different dataset and reference style; not comparable with our .65 on this assignment.
