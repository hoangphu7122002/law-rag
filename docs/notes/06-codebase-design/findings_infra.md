# Findings: experiment infra patterns (2026-11-17)

Source: research subagent. [K] = agent's own knowledge, no source.

1. **Config:** YAML + pydantic v2 (`extra="forbid"`), `--set a.b=1` overrides; no Hydra (chdir magic, multirun override
   inconsistencies #1302/#2946, override grammar cost). tyro if Python configs preferred.
   brentyi.github.io/tyro/goals_and_alternatives · hydra.cc/docs/1.3/upgrades/1.1_to_1.2/changes_to_job_working_dir/
2. **Tracking:** `runs/<ts>_<name>/{config.json, meta.json (git sha+dirty, model revisions, seed, versions), predictions.jsonl, metrics.json}`
   + append-only `results.tsv` (Karpathy autoresearch: commit, metric, status keep/discard/crash, description).
   MLflow/W&B overkill. datacamp.com/tutorial/guide-to-autoresearch
3. **Eval harness:** staged cache — retrieval (`retrieved.jsonl`, key = index + embedder rev + query set) →
   generation (greedy, key = retrieval key + prompt hash + model rev) → metrics. Pin HF revisions. vLLM greedy not
   bit-deterministic across batch sizes [K] → cache is truth. Paired bootstrap 10k, Holm if many comparisons.
   zeroentropy.dev/concepts/statistical-significance · arxiv.org/pdf/2409.03759
4. **LLM backend:** one `openai` client, `base_url` in config; vLLM serve + `ssh -L 8000:localhost:8000`; Modal web_server;
   Mac llama.cpp / mlx_lm.server. Interface `generate(messages, **params) -> text`. Batch jobs may use vLLM `LLM` class.
   modal.com/docs/examples/llm_inference
5. **Remote GPU [K]:** git pull on box, rsync data, tmux/nohup, status.json in run dir, `make sync/fetch`, uv lock both sides.
