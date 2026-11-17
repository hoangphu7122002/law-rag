# Findings: RAG research repo layouts (2026-11-17)

Source: research subagent. Some fetches partial (FlashRAG tree, AutoRAG internals).

| Repo | Key abstraction | Keep | Overkill for solo |
|---|---|---|---|
| FlashRAG (github.com/RUC-NLPIR/FlashRAG) | retriever / generator / refiner classes from config factories; `SequentialPipeline(config).run(dataset)`; yaml + dict override | pipeline = plain class wiring components; config dict overrides yaml | 4 pipeline families, 20+ methods, factory zoo |
| AutoRAG (legacy/, github.com/Marker-Inc-Korea/AutoRAG) | node (stage) × module (candidate) yaml sweep, greedy best per stage | yaml sweep idea; numbered trial dirs + summary.csv | auto-optimizer, dashboard |
| Haystack 2.x (docs.haystack.deepset.ai) | `@component` with typed `run()` → dict; graph pipeline | typed I/O, named components | graph engine, serialization, integrations |
| RAGLAB (arxiv.org/html/2408.11381v1) | fixed seeds/generator/retriever/instructions for fair comparison | change one factor only | trainers, GPU manager |
| cookiecutter-data-science v2 | data/{raw,interim,processed}, notebooks explore only, code in package | immutable raw, one-direction data stages | mkdocs, references/ |

Config guides (livaage repo-layout, agent-tunix, RBC Borealis, anthonyhu research-workflow): separate by rate of change;
base config + small per-experiment overrides; store config with results; Hydra common but heavy for solo.

## Principles
1. One component kind = one small Protocol + dict registry.
2. Typed dataclasses between stages in one `types.py`.
3. Pipeline = plain function/class.
4. One self-describing yaml per experiment over `base.yaml`.
5. Run dir: resolved config, predictions.jsonl, metrics.json, git hash; + summary table.
6. Hold all constant except the factor under test.
7. Data stages immutable, derived artifacts rebuildable by one command.
8. src layout package + thin CLI.
9. Notebooks explore only.
10. Sweep = loop over yamls / small grid expander.
11. Evaluator separate: reads predictions.jsonl, writes metrics.json.
12. Tests on deterministic parts + tiny e2e smoke.
