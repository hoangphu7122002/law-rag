# Findings: datamate codebase audit (m5 ~/Desktop/datamate/datamate, read-only, 2026-11-17)

Size: src 394 .py / 25,084 lines (modules 12.3k, core 5.6k, adapter 4.5k, workflows 2.5k); tests 163 files / 14.3k; docs 164 files (145 research).

## Abstractions
- modules/<group>/<name>/{module.py @register, defaults.yaml = param schema, README.md Reads/Writes, prompts/*.md}; registry.py 48 lines (dict + pkgutil discover).
- Context dataclass (split, shop, question, pack, candidates, answer, extra, trace); pipeline yaml = per_split/per_shop/per_question dicts keyed by step id, `{use:, with:}`; unknown params → ConfigError; per-question failure → abstain + trace.
- adapter/factory.build_adapters(runtime yaml) → llms by role, embedder, index, sandbox; runtime yaml lists vLLM servers {kind, port, gpu, model}; ServerGroup starts them.
- Hydra compose; experiments = `# @package _global_` diff files; compose() returns plain dict.
- Run dir runs/<name>__<confighash>__<runtime>/ {config.yaml, command.txt, metadata.json, predictions.jsonl, traces.jsonl, metrics.json}, append-only → resume; ledger.tsv with strict LedgerRow dataclass.

## Copy
registry + defaults.yaml as schema; step-id pipeline yaml; diff-only experiments; resumable hash-named run dir + strict tsv ledger;
failure isolation per question; runtime yaml with comment header (box, IP, VRAM); AGENTS.md rules (heavy imports inside functions,
cloud dry-run unless --yes, leak rule); GLOSSARY wins over code; `<pkg>_utils.py` naming.

## Avoid
- 106 __init__.py, 10 adapter packages for 4.5k lines, 1–5-line files; nesting like core/tracking/backends/ledger/.
- Files > 500 lines (gate_checks.py 1,045; several 520–660 in scripts).
- 21 near-duplicate runtime yamls + 16 pipeline yamls naming flag combos in filenames; 17 inconsistently named experiments.
- 62 one-off scripts overlapping adapter/launch; 145 research docs, dated ops docs; empty ADR/notes.
- "modules" vs core vs adapter vs workflows unclear; Context.extra grab-bag; Hydra `+`/`~with` friction.
- No ssh/box scripts in this repo (they live in sibling ~/Desktop/datamate/repo or were ad hoc); compute-log by hand.
