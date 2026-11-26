# Weight-space programming agent — brief (arXiv 2611.09382 style)
Goal: raise ROUGE-L of the generator on tail/mid questions by EDITING WEIGHTS directly (no gradient training, no new data). Base = `data/models/dpo_Vp_A2` (dev .6743, len ratio .96) unless the run says otherwise.
Box: `ssh app@100.101.115.108`, repo `~/phunguyen/law_rag`, python `.venv/bin/python` with `PYTHONPATH=src`, one GPU (CUDA_VISIBLE_DEVICES set by the launcher).

## Environment (src/train/weights/wsp_env.py)
- `Env(base, feedback, workdir, budget=50)` loads the model on GPU; `env.model` is a normal HF model you may modify in place (transformer-block weights only; no embeddings / lm_head / architecture changes; stay in bf16 storage).
- `env.eval(n=60)` greedy generation on the first n FEEDBACK prompts → mean rouge_l, len_ratio. Costs 1 of 50 evals (hard budget in `workdir/budget.txt`). `env.logprob(n)` (teacher forcing on gold) is free. `env.reset()` restores the original weights; `env.save(dir)` writes a serve-able checkpoint.
- Feedback set `data/wsp/feedback.jsonl` = train-split tail/mid prompts with gold; decontaminated vs dev/test. DEV AND TEST ARE OFF LIMITS (do not read data/runs, data/processed/split.json dev/test rows, results.tsv). Held-out check is done by the owner afterwards on dev (n=300), then test once.

## Ideas to try (agent decides)
Sparse MLP edits, rank-one updates, per-head attention scaling, scaling the DPO delta in some layers only, directions that shift quote extent (longer/shorter quoting) found from activation differences between good and bad samples. Log every attempt (what, where, #params changed, feedback numbers) in `workdir/log.md`.

## Selection protocol
Pick the final checkpoint by feedback score but require improvement on BOTH the first-60 and a disjoint remaining-60 split (eval on rows via `env.rows` slicing yourself is not allowed beyond the budget). Keeping the original weights is a valid outcome (6/16 paper runs did).
Deliver: `workdir/final/` checkpoint (or "keep original"), `workdir/log.md`, a 10-line summary.
