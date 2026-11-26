# src/train/preference/

preference optimisation: on-policy pair building (pref_data), DPO, GRPO.

Import as `train.preference.<module>`; run as `python -m train.preference.<module>` or `python src/train/preference/<module>.py` with `PYTHONPATH=src`.

| module | what |
|---|---|
| `dpo.py` | DPO on top of an SFT checkpoint with the model's own best-vs-worst samples (src/train/preference/pref_data.py pairs). |
| `grpo.py` | GRPO on top of an SFT checkpoint, verifiable reward vs the course reference answer (no reward model). |
| `pref_data.py` | Preference pairs for DPO from the model's own samples: per train prompt, chosen = the candidate with the highest reward, rejected = |
