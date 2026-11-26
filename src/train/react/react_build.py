"""S1 data for the ReAct student (docs/notes/16-react-training.md): the student's own GOOD trajectories (rejection sampling).

    PYTHONPATH=src python -m train.react_build --runs <run>/trajectories.jsonl ... --out data/train/react_s1 --tools search_law,get_article

Pure helpers: select_s1 (filter + best per question) and stats. main() wires the QA set, decontamination, and gold Điều.
"""
import argparse
import json
from pathlib import Path


def _lookup(source, qid):
    return source(qid) if callable(source) else source.get(qid)


def _tool_call_ids(m: dict) -> list[str]:
    return [c["id"] for c in m.get("tool_calls") or []]


def n_tool_calls(msgs: list[dict]) -> int:
    return sum(len(_tool_call_ids(m)) for m in msgs if m["role"] == "assistant")


def complete(msgs: list[dict]) -> bool:
    """Ends with a non-empty assistant answer (no tool_calls) and every tool call has its tool result."""
    if not msgs or msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls") or not (msgs[-1].get("content") or "").strip():
        return False
    called = {i for m in msgs if m["role"] == "assistant" for i in _tool_call_ids(m)}
    answered = {m.get("tool_call_id") for m in msgs if m["role"] == "tool"}
    return called == answered


def select_s1(rows, reference_of, gold_dieu_of, keep_qids, grade, max_per_question: int = 1) -> list[dict]:
    """Good trajectories of kept questions: final answer grades ok vs the course reference with the gold Điều cited, conversation complete.
    Best rougeL per question first (ties: fewer tool calls), at most `max_per_question`. reference_of / gold_dieu_of: dict or callable by qid."""
    best: dict[str, list[tuple]] = {}
    for i, r in enumerate(rows):
        msgs = r["messages"]
        if r["qid"] not in keep_qids or not complete(msgs):
            continue
        g = grade(msgs[-1]["content"], _lookup(reference_of, r["qid"]) or "", _lookup(gold_dieu_of, r["qid"]) or set())
        if g["ok"]:
            best.setdefault(r["qid"], []).append((-g["rouge"], n_tool_calls(msgs), i, r))
    return [t[3] for qid in best for t in sorted(best[qid], key=lambda t: t[:3])[:max_per_question]]


def _stem(law: str) -> str:
    return law.replace("/", "_")


def retrieved_gold(msgs: list[dict], pairs) -> bool:
    """Did the tool results show a gold Điều? pair 'LAW#N' hits when a tool result has the law stem and a heading 'Điều N.'"""
    text = "\n".join(m.get("content") or "" for m in msgs if m["role"] == "tool")
    return any(_stem(p.rsplit("#", 1)[0]) in text and f"Điều {p.rsplit('#', 1)[1]}." in text for p in pairs or ())


def rewrite_gold(rows, reference_of, gold_pairs_of, keep_qids, max_per_question: int = 1, min_tool_calls: int = 1) -> list[dict]:
    """Style fix for a raw teacher: keep its tool-call turns, replace the final answer by the course reference — but only where the tools
    actually returned a gold Điều (so the reference is grounded in what the model saw). Conversation must be complete and use tools."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        msgs = r["messages"]
        if r["qid"] not in keep_qids or not complete(msgs) or n_tool_calls(msgs) < min_tool_calls:
            continue
        if not retrieved_gold(msgs, _lookup(gold_pairs_of, r["qid"])):
            continue
        ref = _lookup(reference_of, r["qid"]) or ""
        if ref:
            out.setdefault(r["qid"], []).append({**r, "messages": [*msgs[:-1], {"role": "assistant", "content": ref}]})
    return [x for qid in out for x in sorted(out[qid], key=lambda x: n_tool_calls(x["messages"]))[:max_per_question]]


def _tool_names(msgs: list[dict]) -> set[str]:
    return {c["function"]["name"] for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []}


def stats(selected: list[dict], all_rows: list[dict]) -> dict:
    n = len(selected)
    kept_q, total_q = {r["qid"] for r in selected}, {r["qid"] for r in all_rows}
    share = lambda f: sum(map(f, selected)) / n if n else 0.0
    return {"kept": n, "rows_total": len(all_rows), "questions_kept": len(kept_q), "questions_total": len(total_q),
            "coverage": len(kept_q) / len(total_q) if total_q else 0.0,
            "share_with_tool": share(lambda r: n_tool_calls(r["messages"]) > 0),
            "share_with_law_tool": share(lambda r: bool(_tool_names(r["messages"]) & {"get_article", "search_law"}))}


def gold_dieu(pairs) -> set[str]:
    """{'50/2024/TT-NHNN#5', ...} -> {'5'}"""
    return {p.rsplit("#", 1)[1] for p in pairs or ()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="trajectories.jsonl files (data/runs/<run>/trajectories.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="search_law,get_article")
    ap.add_argument("--max-per-question", type=int, default=1)
    ap.add_argument("--final", choices=["student", "gold"], default="student", help="gold: rewrite the final answer of a raw teacher to the reference (needs a gold Điều in the tool results)")
    ap.add_argument("--min-tool-calls", type=int, default=1)
    a = ap.parse_args()

    from corpus.decontam import Decontam
    from corpus.qa import QASet
    from evaluate.retrieval_eval import RetrievalEval
    from generate.react import ALL_TOOLS
    from train.react_data import grade

    train, report = Decontam().filter(QASet().load("train"))
    gold = RetrievalEval().gold
    refs = {q.qid: q.answer for q in train}
    dieu = {q.qid: gold_dieu(gold[q.qid].pairs) if q.qid in gold else set() for q in train}
    rows = [json.loads(line) for p in a.runs for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.final == "gold":
        selected = rewrite_gold(rows, refs, {q.qid: set(gold[q.qid].pairs) if q.qid in gold else set() for q in train}, set(refs), a.max_per_question, a.min_tool_calls)
    else:
        selected = select_s1(rows, refs, dieu, set(refs), grade, a.max_per_question)
    specs = [ALL_TOOLS[n] for n in a.tools.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"qid": r["qid"], "messages": r["messages"], "tools": specs}, ensure_ascii=False) + "\n" for r in selected)
    rep = {"runs": a.runs, "tools": a.tools, "max_per_question": a.max_per_question, "decontam": report, "stats": stats(selected, rows)}
    Decontam.save(rep, out / "report.json")
    print(f"react s1: {len(selected)} trajectories from {len(rows)} rows -> {out}; {rep['stats']}")


if __name__ == "__main__":
    main()
