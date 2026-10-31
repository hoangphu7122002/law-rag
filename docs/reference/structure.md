# Structure

`src/` is on `PYTHONPATH`; nothing is installed. Flow of one question:

```
question ─► retrieve/ ─► passages ─► generate/ (prompt + LLM) ─► answer ─► evaluate/
              ▲
           index/ ◄── corpus/ (load, parse Điều/Khoản, law ids)
```

| Folder | Owns | May import |
|---|---|---|
| `shared/` | config, paths, small types | nothing in `src/` |
| `corpus/` | read raw files, parse structure, QA split | `shared` |
| `index/` | chunking, dense / BM25 / graph builds | `shared`, `corpus` |
| `retrieve/` | ranking at query time | `shared`, `index` |
| `generate/` | prompts, LLM backends, LoRA | `shared` |
| `evaluate/` | metrics, reports | `shared` |
| `cli/` | wiring; one module per make target | everything |

Heavy libraries (torch, faiss, vllm) are imported inside functions so `make test` stays fast.
GPU work runs through `scripts/vendors/modal/`.
