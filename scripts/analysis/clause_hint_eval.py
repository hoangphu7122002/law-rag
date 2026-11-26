"""How often does the lexical clause hint name the khoản the reference quotes? Over dev questions whose reference names "khoản K Điều N", with the cached retrieval of A5_v3b.
    PYTHONPATH=src python scripts/analysis/clause_hint_eval.py   (box)"""
import json
import re
import statistics as st

import polars as pl

from config import load
from generate.clause_hint import clause_hint, split_units
from retrieve.retrieval import Retrieval

REF = re.compile(r"khoản\s+(\d+)\s+Điều\s+(\d+[a-zđ]?)")
cfg = load("configs/exp/A5_v3b.yaml")
final = Retrieval(cfg)._read()
dev = set(json.load(open("data/processed/split.json"))["dev"])
qa = {r["qid"]: r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
n = inpass = 0
hits = {1: 0, 2: 0, 3: 0}
dieu_hits = 0
for qid in sorted(dev & final.keys()):
    m = REF.search(qa[qid]["answer"])
    if not m:
        continue
    gk, gd = m.groups()
    n += 1
    ps = final[qid]
    units = [u for p in ps for u in split_units(p)]
    if any(u.dieu == gd and u.khoan == gk for u in units):
        inpass += 1
    for t in (1, 2, 3):
        h = clause_hint(qa[qid]["question"], ps, top=t)
        if f"khoản {gk} Điều {gd}" in h:
            hits[t] += 1
    if f"Điều {gd}" in clause_hint(qa[qid]["question"], ps, top=3):
        dieu_hits += 1
print(json.dumps({"questions_with_named_khoan": n, "gold_khoan_present_in_context_units": inpass, "top1": hits[1], "top2": hits[2], "top3": hits[3], "top3_dieu_level": dieu_hits}))
