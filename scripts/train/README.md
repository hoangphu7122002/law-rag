# scripts/train/

training helpers: DPO sampling and pair building, teacher repair.

Run from the repo root: `PYTHONPATH=src python scripts/train/<file>.py ...` (scripts are not a package; their tests add this folder to `sys.path` in `tests/unit/scripts/conftest.py`).

| file | what |
|---|---|
| `build_pairs.py` | cands.jsonl -> pairs.jsonl (DPO), parallel over rows (the pure-python ROUGE-L is slow). |
| `pref_sample.py` | Sample candidates for DPO: per train row (src/train/sft/sft_data.py format) one greedy + (n-1) samples at T0.7 from a vLLM endpoint. |
| `teacher_repair.py` | Teacher repair of wrong ReAct trajectories (DeepSeek flash via vibe_call.py). |
