"""Print the trajectory monitor table. Usage: PYTHONPATH=src python3 scripts/ops/traj_monitor.py [--no-tools] [--by split|gold|rouge] FILE.jsonl ..."""
import argparse
import json
from pathlib import Path

import polars as pl

from train.react_build import gold_dieu
from train.react_data import grade
from train.traj_stats import gold_count_group, rouge_band, should_stop, summarize

KEYS = ["n", "completed", "tool_call_rate", "mean_calls", "seen_gold_initial", "good_rate", "mean_rouge", "empty_rate", "loop_rate", "cap_rate", "coverage", "mean_tokens"]


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", choices=["split", "gold", "rouge"], default="gold")
    ap.add_argument("--no-tools", action="store_true", help="run is not meant to use tools")
    a = ap.parse_args()
    qa = {r["qid"]: r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
    gold = {r["qid"]: gold_dieu(r["pairs"]) for r in pl.read_parquet("data/processed/gold.parquet").iter_rows(named=True)}
    for f in a.files:
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["qid"] in qa]
        ref = lambda q: qa[q]["answer"]
        gd = lambda q: gold.get(q, set())
        if a.by == "split":
            group = lambda q: qa[q].get("split", "?")
        elif a.by == "gold":
            group = lambda q: gold_count_group(len(gd(q)))
        else:  # rouge band of the best-known single score needs grading: use the first row's answer per qid
            first = {}
            for r in rows:
                first.setdefault(r["qid"], r)
            sc = {q: (grade(r["messages"][-1].get("content") or "", ref(q), gd(q))["rouge"]) for q, r in first.items()}
            group = lambda q: rouge_band(sc[q])
        s = summarize(rows, ref, gd, grade, group_of=group)
        print(f"== {f}")
        print("group\t" + "\t".join(KEYS))
        for name, g in [("all", s), *s.get("groups", {}).items()]:
            print(f"{name}\t" + "\t".join(fmt(g.get(k, "")) for k in KEYS))
        print("calls hist:", s.get("calls_hist"), "| rouge by calls:", s.get("rouge_by_calls"))
        print("STOP:", should_stop(s, expects_tools=not a.no_tools) or "ok")


if __name__ == "__main__":
    main()
