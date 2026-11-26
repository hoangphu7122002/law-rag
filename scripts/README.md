# scripts/

Entry points and diagnostics, grouped by pipeline stage. Reusable logic lives in `src/`; scripts stay thin.

| folder | purpose |
|---|---|
| [data/](data/README.md) | build data: CPT corpus, augmentation, external text, synthetic QA, SFT subsets |
| [train/](train/README.md) | DPO sampling and pair building, teacher repair |
| [analysis/](analysis/README.md) | diagnostics / reports on finished runs (tail-mid, cited group, coverage, failure stages, MBR) |
| [serve/](serve/README.md) | serving, load test, demo, API evaluation |
| [ops/](ops/README.md) | box watchdog, trajectory monitor, ReAct launchers |

Old flat paths (`scripts/<name>.py`) are listed in [../docs/MOVED.md](../docs/MOVED.md).
