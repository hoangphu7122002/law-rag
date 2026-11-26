"""Offline test of an extent post-process on existing dev predictions (no GPU): complete the quoted unit. Answer lines are aligned to the lines of the passage they quote
(token F1 >= .8); when some but not all lines of a unit (a khoản line with its a)/b)/... items) were quoted, the missing lines of that unit are added in passage order.
Reports mean ROUGE-L before/after overall and for the answers that changed, with a paired bootstrap.

    PYTHONPATH=src python scripts/analysis/postprocess_extent.py data/runs/A5_v3b__2e66c86a   (box)
"""
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

import polars as pl

from config import load
from evaluate.metrics import normalize_text, rouge_l_score
from retrieve.retrieval import Retrieval

UNIT_START = re.compile(r"^\s*(\d+[\.\)]|Điều\s+\d+)")
ITEM = re.compile(r"^\s*([a-zđ]|[ivx]+)\)")


def tokens(s: str) -> list[str]:
    return normalize_text(s).split()


def f1(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    ca, cb = {}, {}
    for t in a:
        ca[t] = ca.get(t, 0) + 1
    for t in b:
        cb[t] = cb.get(t, 0) + 1
    common = sum(min(v, cb.get(t, 0)) for t, v in ca.items())
    if not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def strip_bullet(line: str) -> str:
    return re.sub(r"^\s*[-•*]\s*", "", line)


def units(lines: list[str]) -> list[int]:
    """unit id per passage line: a numbered line starts a new unit, a)/b) items stay with the unit they follow."""
    out, u = [], 0
    for i, ln in enumerate(lines):
        if i and UNIT_START.match(ln) and not ITEM.match(ln):
            u += 1
        out.append(u)
    return out


def complete(answer: str, passages: list[str], thr: float = 0.8, min_cov: float = 0.0, max_missing: int = 99) -> str:
    lines = answer.split("\n")
    if len(lines) < 2:
        return answer
    head, body = lines[0], [ln for ln in lines[1:] if ln.strip()]
    btok = [tokens(strip_bullet(ln)) for ln in body]
    best = None
    for p in passages:
        pl_ = [ln for ln in p.split("\n") if ln.strip()]
        ptok = [tokens(ln) for ln in pl_]
        hits = {}
        for j, t in enumerate(btok):
            cand = max(((f1(t, pt), i) for i, pt in enumerate(ptok)), default=(0, -1))
            if cand[0] >= thr:
                hits[cand[1]] = j
        if best is None or len(hits) > len(best[2]):
            best = (pl_, ptok, hits)
    if not best or not best[2]:
        return answer
    pl_, ptok, hits = best
    uid = units(pl_)
    add = set()
    for u in {uid[i] for i in hits}:
        members = [k for k in range(len(pl_)) if uid[k] == u]
        got = [k for k in members if k in hits]
        if len(got) / len(members) >= min_cov and len(members) - len(got) <= max_missing:
            add.update(members)
    if add <= set(hits):
        return answer
    keep = sorted(set(hits) | add)
    # keep unmatched (paraphrased) answer lines in place at the end only if they were not aligned
    unmatched = [ln for j, ln in enumerate(body) if j not in set(hits.values())]
    new_body = [pl_[i] for i in keep]
    return "\n".join([head, *new_body, *unmatched])


run = Path(sys.argv[1])
MIN_COV = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
MAX_MISS = int(sys.argv[3]) if len(sys.argv) > 3 else 99
cfg = load("configs/exp/A5_v3b.yaml")
final = Retrieval(cfg)._read()
qa = {r["qid"]: r["answer"] for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
pred = {}
for line in open(run / "predictions.jsonl", encoding="utf-8"):
    r = json.loads(line)
    pred[r["qid"]] = r["answer"]
before, after, changed = {}, {}, 0
for qid, a in pred.items():
    ps = [p.text for p in final.get(qid, [])]
    b = complete(a, ps, min_cov=MIN_COV, max_missing=MAX_MISS)
    before[qid], after[qid] = rouge_l_score(a, qa[qid]), rouge_l_score(b, qa[qid])
    changed += b != a
d = [after[q] - before[q] for q in pred]
rng = random.Random(0)
boot = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(3000))
print(f"n={len(d)} changed={changed} before={st.mean(before.values()):.4f} after={st.mean(after.values()):.4f} diff={st.mean(d):+.4f} 95% CI [{boot[75]:+.4f},{boot[2924]:+.4f}]")
ch = [q for q in pred if after[q] != before[q]]
if ch:
    print(f"changed answers: before {st.mean(before[q] for q in ch):.4f} after {st.mean(after[q] for q in ch):.4f}; improved {sum(after[q] > before[q] for q in ch)} worse {sum(after[q] < before[q] for q in ch)}")
