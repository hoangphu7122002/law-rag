# configs/queues/react/ (10 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/react/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `pass1_g0.txt` | Teacher data pass 1 (no tools, retrieval profile): Qwen3.5-9B answers dev then train from profile + pipeline top-5 (2 shots): greedy, then T 0.7. GPU0; retrieval via embed :8101 / rerank :8201 on GPU1; KG data/kg/kg.sqlite. Graded |
| `pass1b_g0.txt` | Probe: 9B with tools (search_law/get_article/kg_related), dev. Needs the tool-call parser, so the server is restarted with --tools. Runs after pass1_g0 finishes. |
| `pass2_g1.txt` | Pass 2: 9B with tools forced (min_tool_calls 1) on dev then train, profile + top-5 in the first prompt, shared tools endpoint :8300. Graded afterwards; failing questions go to DeepSeek repair. |
| `probe_g1.txt` | 9B tool probe on the idle GPU1 (beside embed/rerank), dev: does the base 9B call tools when given profile + top-5? |
| `react_compact_g1.txt` | Compact tool prompt (drops the duplicated per-tool prose list) on SFT v0, T0 tools, dev_clean; control = A7_react__bbf154af (full prompt, .5709). After the ReAct reruns on GPU1. |
| `react_g0.txt` | ReAct baseline on RAW Qwen3.5-2B (dev_clean 166): (a) no tools (plain raw model, same prompt), (b) ReAct + 5 initial passages, (c) ReAct with no initial passages. |
| `react_rerun_g1.txt` | rerun of the two ReAct arms that died unfinished (T3 = +KG +web: 58/166 rows; no-initial-passages arm: 0 rows) after GPU1's GRPO c6f queue; raw Qwen3.5-2B, dev_clean |
| `tools_g0.txt` | ReAct tool-set arms T0..T3 on the RAW Qwen3.5-2B, dev_clean (DRAFT, not queued: T2/T3 need TINYFISH_API_KEY on the box + the user's go-ahead for web calls) |
| `tools_kg_g1.txt` | GPU1, after the ReAct baseline on the RAW model (queue-1119-180219): |
| `tools_web_g0.txt` | T2 (+web) then T3 (+KG +web) on the RAW Qwen3.5-2B, dev_clean, TinyFish BUILDER (/bin/zsh.005/query, hard cap $6 over all runs, 150/min). After the ReAct baseline queue. |
