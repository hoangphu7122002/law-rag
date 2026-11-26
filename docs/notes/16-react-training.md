# Training the ReAct student (2026-11-19) — plan

Goal: a <4B student that uses search_law / get_article / kg_related better than the raw model (dev_clean rougeL: raw ReAct .411, SFT v0 + ReAct .571, plain RAG v0 .583, v3b MBR .658).
Upper bound is small: tools matter for the ~10 % of questions whose gold Điều is not in the initial top-5 (≈ +.03 rougeL if fixed).

Arms (each against the non-agentic control; accept only with a paired-bootstrap gain on dev_clean/dev_strict):
1. **Haiku teacher-fix (datamate S1/S2).** Student rollouts (v0/v3b, K samples, T .7) → grade (src/train/react/react_data.py: rougeL ≥ .6 and cites gold Điều).
   S1 = good trajectories; S2 = wrong ones repaired by the teacher (Haiku 5.5 low, paced batches of ~20 via the Agent tool, memory haiku-pacing) from the first bad step
   (gold Điều never reached the model → next turn = get_article(law, dieu) or search; later turns = answer). A repair is kept only if re-running its tool call
   returns the gold Điều and the final answer grades ok without copying the reference (13-gram guard). Rows: one per assistant turn, loss on completions only.
2. **On-policy distillation, Qwen3.5-9B teacher** (docs/research/on-policy-distill/findings.md): TRL 1.15 `DistillationTrainer` (tools=, always on-policy, reverse KL with beta=1).
   Gate first: 9B ReAct / RAG on dev_clean must beat SFT v0 (.58), else skip.
3. Fallback: self-distillation with the gold Điều in the teacher prompt; multi-turn GRPO.

datamate warnings carried over: a loop SFT on 1,764 trajectories scored below one-shot; untrained-model self-traces are ~13 % usable; ≤3B students do not always profit from large-teacher
distillation (mix sources); noise ±.01 between identical runs → paired tests only.
