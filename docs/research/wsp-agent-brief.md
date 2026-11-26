# Weight-space programming agent — brief (follows arXiv 2611.09382, Hu et al. 2026)
Goal: raise ROUGE-L of the generator on tail/mid questions by EDITING WEIGHTS directly. Base = `data/models/dpo_Vp_A2` (dev .6743, len ratio .96, quote extent / which-khoản ambiguity is the bottleneck: reference answers quote the right provision, model under/over-quotes or cites another Điều).
Box: `ssh app@100.101.115.108`, repo `~/phunguyen/law_rag`, `PYTHONPATH=src .venv/bin/python`, one GPU. Original checkpoint is never modified.

## Rules (paper-faithful, relaxed where the user asked)
- No gradient training, no running the model outside the forward tool. Edit by writing weights: `θ ← bf16(θ + Δθ)` with Δθ computed in float32 by your own Python.
- User relaxation: ANY layer/tensor may be edited (attention, MLP, norms, even embeddings / lm_head) — not limited to LoRA-style or block weights. Architecture and tokenizer fixed. Whole-tensor constant scaling is out of scope.
- Forward tool = `Env` (src/train/weights/wsp_env.py): **50 successful requests** (`generate` = G, `tf` = TF; a request may hold many prompts); **≤300 Python analyses** (call `env.analysis()` per step). G can run original, edited or both in one request and returns the paired difference with SE. TF returns log-probs, top-k, hidden states, MLP activations (no score).
- Data: feedback set `data/wsp/feedback.jsonl` (train-split tail/mid prompts + gold, decontaminated). Rows 0–149 = discovery, rows 150–299 = fresh confirmation. DEV AND TEST ARE OFF LIMITS (do not open data/runs, results.tsv, split dev/test rows). We check dev (n=300) afterwards, test once.

## Protocol (as in the paper)
1. Diagnose first: G on the original (what fails: length ratio, truncations, wrong khoản, refusals); TF on gold vs the model's own answers to localise (which layers/heads/channels write the over/under-quote or the wrong-Điều decision).
2. Ablation scans on a small batch (scale/zero heads, ablate channels, project out directions) — keep a table of every request (what, where, #changed values via `env.n_changed()`, rouge_l, paired SE).
3. Build a direction from contrasts (e.g. good-extent vs bad-extent answers, or "cites gold Điều" vs "cites other Điều": mean residual/MLP-readout difference), apply as a rank-one / single-column / head edit.
4. Controls: reversed direction, equal-norm random orthogonal direction, dose sweep (several strengths).
5. Keep an edit only if the gain on fresh confirmation rows (150–299, `weights="both"`) is clearly above the paired SE (> 2 SE) with regressions not > rescues. Keeping the original weights is a valid outcome (6/16 paper searches did).
Deliver: `workdir/final/` (via `env.save`) or "keep original", `workdir/log.md` (request table), 10-line summary.
