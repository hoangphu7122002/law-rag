# src/train/react/

ReAct student: trajectory data, teacher repair, SFT, trajectory statistics.

Import as `train.react.<module>`; run as `python -m train.react.<module>` or `python src/train/react/<module>.py` with `PYTHONPATH=src`.

| module | what |
|---|---|
| `react_build.py` | S1 data for the ReAct student (docs/notes/16-react-training.md): the student's own GOOD trajectories (rejection sampling). |
| `react_data.py` | Trajectory data for the ReAct student, datamate style (docs/notes/16-react-training.md): S1 = the student's own good trajectories, |
| `react_fix.py` | Teacher-fix tooling for ReAct training data (docs/notes/16-react-training.md, S2): pure helpers, the teacher (Haiku) and the tools |
| `react_sft.py` | LoRA SFT of the ReAct student on its own good trajectories (src/train/react/react_build.py, datamate S1), loss on the ASSISTANT turns only |
| `repair_run.py` | Teacher-repair runner core (docs/research/teacher-data-plan.md, stage 3): pure and testable; the teacher (prompt -> text) and the tools |
| `traj_stats.py` | Trajectory monitor for ReAct rollouts (teacher-data-plan Stage 1b). Pure functions, no I/O. |
