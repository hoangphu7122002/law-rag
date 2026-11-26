"""Cascade: MBR only for the least confident share of questions, greedy for the rest.

    PYTHONPATH=src python scripts/analysis/cascade_stats.py <greedy_run> <mbr_run> [--exp configs/exp/A6_mbrv3b.yaml]     (box, dev)
Signal = reranker margin (top-1 minus top-2 score from the cached retrieval): low margin = unsure -> MBR. Cost in generations per question
(greedy 1, MBR8 8). Prints rougeL of the cascade per share of questions sent to MBR.
"""
import json
import statistics as st
import sys


def cascade_curve(rows: list[dict], fractions=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0), mbr_cost: int = 8) -> dict:
    order = sorted(rows, key=lambda r: r["signal"])
    out = {}
    for f in fractions:
        k = round(f * len(order))
        out[f] = {"rouge": round(st.mean([r["mbr"] for r in order[:k]] + [r["greedy"] for r in order[k:]]), 4),
                  "cost": round((k * mbr_cost + (len(order) - k)) / len(order), 2)}
    return out


if __name__ == "__main__":
    from config import load
    from corpus import QASet
    from evaluate.metrics import rouge_l_score
    from retrieve import Retrieval

    g_run, m_run = sys.argv[1], sys.argv[2]
    exp = sys.argv[sys.argv.index("--exp") + 1] if "--exp" in sys.argv else "configs/exp/A6_mbrv3b.yaml"
    cfg = load(exp, ["split=dev"], None)
    qs = QASet().load("dev")
    ref = {q.qid: q.answer for q in qs}
    passages = Retrieval(cfg).run(qs)
    pred = lambda run: {json.loads(l)["qid"]: json.loads(l)["answer"] for l in open(f"data/runs/{run}/predictions.jsonl")}  # noqa: E731
    g, m = pred(g_run), pred(m_run)
    rows = []
    for q in qs:
        s = [p.score for p in passages[q.qid]]
        if q.qid in g and q.qid in m and len(s) > 1:
            rows.append({"signal": s[0] - s[1], "greedy": rouge_l_score(g[q.qid], ref[q.qid]), "mbr": rouge_l_score(m[q.qid], ref[q.qid])})
    print(len(rows), "questions; mean greedy", round(st.mean(r["greedy"] for r in rows), 4), "mean mbr", round(st.mean(r["mbr"] for r in rows), 4))
    print(json.dumps(cascade_curve(rows)))
    rs = sorted(rows, key=lambda r: r["signal"])
    for name, part in (("low-margin third", rs[:len(rs) // 3]), ("mid third", rs[len(rs) // 3:2 * len(rs) // 3]), ("high-margin third", rs[2 * len(rs) // 3:])):
        print(name, "MBR gain", round(st.mean(r["mbr"] - r["greedy"] for r in part), 4))
