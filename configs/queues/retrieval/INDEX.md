# configs/queues/retrieval/ (20 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/retrieval/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `a2e.txt` | one job per line, run one after another on the box: cli box queue configs/queue_a2e.txt |
| `a3.txt` | A3 decoding sweep, dev. serve lines swap the LLM on GPU1 (port 8001). |
| `a3p.txt` | A3 raw base of Vi-Qwen2 (fine-tune gain reference), dev |
| `a3r2.txt` | A3 round 2 on Q3 (Qwen3.5-2B thinking T0.6), dev |
| `ck62_g1.txt` | Chunking ablation Stage 1 (Track B, GPU1 embed :8101 + rerank :8201): per variant build the index, retrieval-only dev + test (dense top-50 -> rerank top-5, k=10), delete the index. Control = the existing index 4ac2a75c (retr-rr50- |
| `ck63_g1.txt` | Chunking Stage 1, rest after the transient-scope clash: khoan1000 test (its dev run is already running), sem1000 dev+test, sent1000 test (index rebuilt: the first one was deleted before the test run could start). |
| `ck69_lowprio.txt` | Chunking Stage 1b, LOW PRIORITY (runs only while no LLM / training job is active on the box): recursive 512-character chunks (~2.2 M nodes, bigger than the semantic-1000 build that was OOM-killed at 32 GB; cap now 46 GB). Retrieva |
| `expand_eval_g1.txt` | Gate + widen experiment (user-requested): dev300 and TEST ROUGE-L of PiSSA(C) with the module off (k=101 base top-5) and on (102 gate->top10, 103 gate->top15, 104 always top15, 105 oracle widen). Retrieval is cached (scripts/analy |
| `full68_g1.txt` | Full corpus index (Track B, GPU1 embed :8101 + rerank :8201): sources + web_new_laws (10 fetched cited laws); retrieval-only dev + test, k=10, compare with the control index 4ac2a75c via scripts/analysis/retr_compare.py |
| `full_g0.txt` | Pre-declared top candidates for the full pass (chosen from dev_clean/dev_strict, not from test): MBR8 on rl_c2, MBR8 on v3b+2 shots (test) |
| `full_g1.txt` | MBR16 on v0, test (dev row comes from queue-1119-172845) |
| `fullart_g0.txt` | Full-Điều context test (no chunk cutting): caches 201 (first 5 distinct articles of the top-15 chunks, whole article), 202 (first 8), 203 (articles of the top-5 chunks only). dev300 only. D3c rsLoRA was trained with full gold arti |
| `k10_g0.txt` | k=10 evals on dev (GPU0, port 8000), after the A4 GPU0 queue. Models trained on k=5 data. |
| `llmrr66_g0.txt` | LLM re-ranking of the top-10 (GLM-5.3 low, DeepSeek-V4 flash) -> dpo_a2 greedy on dev300 with the helper's order in the context (exploratory upper bound). Reference: same model with the reranker's order .6637 (A5_v3b__35ff9780). l |
| `retr59.txt` | Retrieval-only ablation (no LLM): dense vs reranker (candidates 20/50/100) vs BM25 vs hybrid(+rerank), dev and test, k=10. Output: data/runs/logs/retr-<variant>-<split>.log (one JSON line: doc@k / Dieu@k / MRR). |
| `retr61.txt` | re-run of the hybrid variants that died silently (OOM suspected); one at a time |
| `rr2_g0.txt` | (no description line) |
| `rr3_g0.txt` | (no description line) |
| `rr4_g1.txt` | GPU1: v3b + fine-tuned reranker rr_v0 (local engine), dev only. Waits for the v3 eval queue. |
| `rr_g0.txt` | (no description line) |
