"""Paired bootstrap of per-question retrieval hits between two retrieval caches (control vs a chunking variant).

    PYTHONPATH=src python scripts/retr_compare.py --split dev [--k 5] [--level art] --variant "chunk.split=khoan" --variant "chunk.size=2000"
The control is the default config (the cached index 4ac2a75c); the variant adds the --variant overrides. Both caches must exist (run
`cli run ... --retrieval-only` first). level art = gold (law, Điều) pair among the top k; doc = gold law among the top k.
"""
import argparse
import json
from pathlib import Path


def per_question_hits(cache: Path, gold: dict, k: int, level: str = "art") -> dict[int, float]:
    from evaluate import RetrievalEval

    out = {}
    for line in Path(cache).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        g = gold.get(r["qid"])
        target = (g.pairs if level == "art" else g.docs) if g else set()
        if not target:
            continue
        keys = [RetrievalEval.passage_keys(p) for p in r["passages"][:k]]
        ids = [f"{d}#{a}" if a else f"{d}#?" for d, a in keys] if level == "art" else [d for d, _ in keys]
        out[r["qid"]] = float(any(i in target for i in ids))
    return out


def main() -> None:
    from config import load
    from corpus.gold import GoldSet
    from evaluate import PairedBootstrap
    from retrieve.retrieval import CACHE

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--level", default="art")
    ap.add_argument("--variant", action="append", default=[], help="override KEY=VALUE of the variant (repeatable)")
    a = ap.parse_args()
    base = [f"split={a.split}", "retrieve.k=10"]
    gold = GoldSet().load()
    caches = {n: CACHE / f"{load('configs/exp/A5_v3b.yaml', base + extra).retrieval_key()}.jsonl" for n, extra in (("control", []), ("variant", a.variant))}
    hits = {n: per_question_hits(p, gold, a.k, a.level) for n, p in caches.items()}
    res = PairedBootstrap().compare(hits["control"], hits["variant"])
    print(json.dumps({"split": a.split, "level": a.level, "k": a.k, "control": round(sum(hits["control"].values()) / len(hits["control"]), 4),
                      "variant": round(sum(hits["variant"].values()) / len(hits["variant"]), 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}}))


if __name__ == "__main__":
    main()
