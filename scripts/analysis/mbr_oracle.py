"""Generator ceiling: sample the MBR candidates per dev question, keep them all, score greedy / MBR pick / best-of-n vs the reference.

    PYTHONPATH=src python scripts/analysis/mbr_oracle.py configs/exp/A6_mbrv3b.yaml out.jsonl --set generate.shots=2 --set split=dev_clean [--backend l40s]
Needs the experiment's LLM endpoint (box); retrieval comes from the cache. Dev only (never the test split).
"""
import argparse
import asyncio
import json
import statistics as st

from config import load
from corpus import QASet
from evaluate.metrics import rouge_l_score
from generate import Generator
from retrieve import Retrieval


def summarize(rows: list[dict]) -> dict:
    """rows: {"cands": [...], "ref": str}; cands[0] is the greedy one. Mean ROUGE-L of greedy, MBR pick, mean sample, best-of-n."""
    def mean(f):
        return round(st.mean(f(r) for r in rows), 4)

    out = {"n": len(rows),
           "greedy": mean(lambda r: rouge_l_score(r["cands"][0], r["ref"])),
           "mbr": mean(lambda r: rouge_l_score(Generator.mbr_pick(r["cands"]), r["ref"])),
           "mean_sample": mean(lambda r: st.mean(rouge_l_score(c, r["ref"]) for c in r["cands"]))}
    for n in (2, 4, len(rows[0]["cands"])):
        out[f"best_of_{n}"] = mean(lambda r: max(rouge_l_score(c, r["ref"]) for c in r["cands"][:n]))
    out["len_ref"] = mean(lambda r: len(r["ref"]))
    out["len_mbr"] = mean(lambda r: len(Generator.mbr_pick(r["cands"])))
    return out


async def collect(cfg, questions, passages) -> list[dict]:
    g = Generator(cfg.generate, cfg.backend)
    sem = asyncio.Semaphore(cfg.generate.concurrency)

    async def one(q):
        async with sem:
            return {"qid": q.qid, "ref": q.answer, "cands": await g.candidates(q.text, passages[q.qid])}

    return await asyncio.gather(*(one(q) for q in questions))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("out")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    cfg = load(a.exp, a.set, a.backend)
    assert not cfg.split.startswith("test"), "dev only"
    questions = QASet().load(cfg.split, cfg.limit)
    rows = asyncio.run(collect(cfg, questions, Retrieval(cfg).run(questions)))
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(json.dumps(summarize(rows)))


if __name__ == "__main__":
    main()
