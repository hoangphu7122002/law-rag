# scripts/ops/

box operations: watchdog, trajectory monitor, ReAct launchers.

Run from the repo root: `PYTHONPATH=src python scripts/ops/<file>.py ...` (scripts are not a package; their tests add this folder to `sys.path` in `tests/unit/scripts/conftest.py`).

| file | what |
|---|---|
| `box_watchdog.sh` | User-level guard for the shared box (no root: no earlyoom, no hardware watchdog). Run it in tmux `lawrag-watchdog`. |
| `react_g1.sh` | ReAct SFT from a raw teacher (docs/notes/16-react-training.md, "G1"): build from the raw Qwen3.5-9B trajectories on TRAIN (tool turns kept, |
| `react_s1.sh` | ReAct rejection-sampling SFT (S1): build from the seed-1 T0 trajectories of the train split, then LoRA-SFT from SFT v0. Run on the box. |
| `traj_monitor.py` | Print the trajectory monitor table. Usage: PYTHONPATH=src python3 scripts/ops/traj_monitor.py [--no-tools] [--by split|gold|rouge] FILE.jsonl ... |
