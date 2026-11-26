"""Where do the low-scoring questions fail? Stage of the first failure, per question of a run:

    PYTHONPATH=src python scripts/analysis/low_score_analysis.py data/runs/<run> [--low 0.4]
stages: reference cites no article | gold article not in the passages | retrieved, not cited | retrieved and cited
(passages = the retrieval cache of the run; gold = corpus.gold pairs "<law key>#<Điều>"; "cited" = the answer names a gold Điều number.)
"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path

from evaluate.faithfulness import dieu_numbers


def failure_stage(gold_pairs: set[str], passage_pairs: set[str], answer: str) -> str:
    if not gold_pairs:
        return "reference cites no article"
    if not gold_pairs & passage_pairs:
        return "gold article not in the passages"
    gold_numbers = {int(p.rsplit("#", 1)[1]) for p in gold_pairs if p.rsplit("#", 1)[1].isdigit()}
    return "retrieved and cited" if gold_numbers & dieu_numbers(answer) else "retrieved, not cited"


def summarize_stages(rows: list[tuple[str, float]], low: float = 0.4) -> dict:
    by = collections.defaultdict(list)
    for stage, score in rows:
        by[stage].append(score)
    total_low = sum(1 for _, s in rows if s < low) or 1
    out = {}
    for stage, v in by.items():
        n_low = sum(1 for s in v if s < low)
        out[stage] = {"n": len(v), "mean": round(st.mean(v), 3), "low": n_low, "low_share": round(n_low / total_low, 3)}
    return out


def main() -> None:
    from corpus.gold import GoldSet
    from evaluate import RetrievalEval

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--low", type=float, default=0.4)
    a = ap.parse_args()
    run = Path(a.run)
    gold = GoldSet().load()
    cache, _ = RetrievalEval.cache_of(run)
    passages = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        passages[r["qid"]] = r["passages"]
    scores = {r["qid"]: r["rouge_l"] for r in json.load(open(run / "metrics.json"))["rows"]}
    rows = []
    for line in open(run / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        g = gold.get(p["qid"])
        pairs = {f"{k}#{art}" for k, art in (RetrievalEval.passage_keys(x) for x in passages.get(p["qid"], [])) if art}
        rows.append((failure_stage(set(g.pairs) if g else set(), pairs, p["answer"]), scores[p["qid"]]))
    s = summarize_stages(rows, a.low)
    print(json.dumps({"run": run.name, "n": len(rows), "low_threshold": a.low, "stages": s}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
