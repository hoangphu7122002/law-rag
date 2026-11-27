# configs/queues/

Job queues for the GPU box (`PYTHONPATH=src python -m cli box queue configs/queues/<group>/<file>.txt [--parallel] [--gpu N]`). One file = one lane (`_g0` / `_g1` = GPU). DSL: `sh NAME cmd`, `wait COND`, `do CMD`, `serve MODEL --gpu N`, or an experiment yaml line (see `src/ops/box.py`). The box keeps its own generated copy of a launched queue under `data/runs/logs/queue-*.sh`, so moving a file here never affects a running queue.

| folder | what |
|---|---|
| [sft/](sft/INDEX.md) | SFT runs and data builds: v0..v3b, C/D/D2/D3 data, variants (PiSSA, rsLoRA...), epochs, seeds, learning curves |
| [dpo/](dpo/INDEX.md) | on-policy DPO rounds, ablations (beta, cite weight), replications, DPO on V_pissa |
| [cpt/](cpt/INDEX.md) | continued pretraining data and SFT-from-CPT |
| [rl/](rl/INDEX.md) | GRPO / RL runs |
| [react/](react/INDEX.md) | ReAct / tools / web agents |
| [retrieval/](retrieval/INDEX.md) | retrieval, chunking ablations, reranker, index and context experiments |
| [decoding/](decoding/INDEX.md) | MBR / decoding experiments |
| [eval/](eval/INDEX.md) | dev / test evaluations of finished models, order sensitivity |
| [final/](final/INDEX.md) | final pre-declared chain, thinking SFT, 4B / 9B scaling lanes |
| [night/](night/INDEX.md) | overnight autopilot lanes of the 2026-11-2x campaigns (59 files, numbered) |

Add new queues to the right folder and keep the first line a `# description` comment (INDEX.md is generated from it).
