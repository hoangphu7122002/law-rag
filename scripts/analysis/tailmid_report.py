"""Per-run tail/mid report: ROUGE-L by failure stage (low_score_analysis stages, retrieved-not-cited split into 'cites another retrieved Điều' / 'cites none'),
refusal rate, answer length vs reference, cited-Điều match. Prints one markdown table row per run.

    PYTHONPATH=src python scripts/analysis/tailmid_report.py data/runs/A5_v3b__127c207b [more runs]   (run on the box)
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import polars as pl

from corpus.gold import GoldSet
from evaluate.faithfulness import dieu_numbers
from evaluate.retrieval_eval import RetrievalEval

REFUSAL = re.compile(r"(ngữ cảnh[^.]{0,60}(không (đề cập|có thông tin|nêu)|chưa tìm thấy)|chưa tìm thấy quy định|không thể trả lời)", re.I)
STAGES = ["gold not in passages", "ref cites no article", "retrieved, cites other Điều", "retrieved, cites none", "retrieved and cited"]


def report(run: Path, gold, qa) -> dict:
    cache, _ = RetrievalEval.cache_of(run)
    passages = {json.loads(l)["qid"]: json.loads(l)["passages"] for l in cache.read_text(encoding="utf-8").splitlines()}
    m = json.load(open(run / "metrics.json"))
    scores = {r["qid"]: r["rouge_l"] for r in m["rows"]}
    by = {s: [] for s in STAGES}
    refusals = 0
    ratio, n = [], 0
    for line in open(run / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        n += 1
        g = gold.get(p["qid"])
        ref = qa[str(p["qid"])]["answer"] or ""
        refusals += bool(REFUSAL.search(p["answer"][:300]))
        ratio.append(len(p["answer"]) / max(len(ref), 1))
        pairs = {f"{k}#{a}" for k, a in (RetrievalEval.passage_keys(x) for x in passages.get(p["qid"], [])) if a}
        gp = set(g.pairs) if g else set()
        if not gp:
            st_ = "ref cites no article"
        elif not gp & pairs:
            st_ = "gold not in passages"
        else:
            gn = {int(x.rsplit("#", 1)[1]) for x in gp if x.rsplit("#", 1)[1].isdigit()}
            cited = dieu_numbers(p["answer"])
            if gn & cited:
                st_ = "retrieved and cited"
            else:
                other = {x for x in pairs if x.rsplit("#", 1)[1].isdigit() and int(x.rsplit("#", 1)[1]) in cited}
                st_ = "retrieved, cites other Điều" if other else "retrieved, cites none"
        by[st_].append(scores[p["qid"]])
    return {"run": run.name, "n": n, "rougeL": round(st.mean(scores.values()), 4), "refusal%": round(100 * refusals / n, 1), "len_ratio": round(st.median(ratio), 2),
            **{s: (len(v), round(st.mean(v), 3) if v else None) for s, v in by.items()}}


def main() -> None:
    gold = GoldSet().load()
    qa = {str(r["qid"]): r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
    rows = [report(Path(a), gold, qa) for a in sys.argv[1:]]
    print("| run | n | rougeL | refusal% | len ratio | " + " | ".join(STAGES) + " |")
    print("|" + "---|" * (5 + len(STAGES)))
    for r in rows:
        print(f"| {r['run']} | {r['n']} | {r['rougeL']} | {r['refusal%']} | {r['len_ratio']} | " + " | ".join(f"{r[s][0]} / {r[s][1]}" for s in STAGES) + " |")


if __name__ == "__main__":
    main()
