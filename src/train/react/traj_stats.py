"""Trajectory monitor for ReAct rollouts (teacher-data-plan Stage 1b). Pure functions, no I/O."""
from collections import Counter, defaultdict
from statistics import mean

from train.react.react_build import n_tool_calls
from train.react.react_data import _seen

CHARS_PER_TOKEN = 3.5


def _final(msgs):
    return msgs[-1] if msgs and msgs[-1]["role"] == "assistant" else None


def _answer(msgs) -> str | None:
    """Final assistant text, or None when the rollout has no final answer (empty, tool call, or not assistant)."""
    last = _final(msgs)
    if last is None or last.get("tool_calls"):
        return None
    text = (last.get("content") or "").strip()
    return text or None


def _is_empty(msgs) -> bool:
    last = _final(msgs)
    return last is not None and not last.get("tool_calls") and not (last.get("content") or "").strip()


def _has_loop(msgs) -> bool:
    seen = Counter((c["function"]["name"], c["function"].get("arguments")) for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or [])
    return any(v > 1 for v in seen.values())


def _bucket(k: int) -> str:
    return "3+" if k >= 3 else str(k)


def gold_count_group(n: int) -> str:
    return "3+" if n >= 3 else str(n)


def rouge_band(r: float) -> str:
    return "low" if r < 0.5 else "mid" if r < 0.8 else "high"


def _rate(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _summary(rows, reference_of, gold_dieu_of, grade) -> dict:
    n = len(rows)
    out = {"n": n}
    if not n:
        return out
    calls, done, empty, loops, caps, rouges, oks, seen, tokens = [], [], [], [], [], [], [], [], []
    by_calls = defaultdict(list)
    good_q = set()
    for r in rows:
        m, qid = r["messages"], r["qid"]
        k = n_tool_calls(m)
        gd = gold_dieu_of(qid) or set()
        ans = _answer(m)
        calls.append(k)
        done.append(ans is not None)
        empty.append(_is_empty(m))
        loops.append(_has_loop(m))
        caps.append(k >= 3 and ans is None)
        if gd:
            seen.append(any(_seen(x.get('content'), gd) for x in m[:2] if x['role'] == 'user'))
        tokens.append(sum(len(x.get("content") or "") for x in m) / CHARS_PER_TOKEN)
        if ans is None:
            oks.append(False)
            continue
        g = grade(ans, reference_of(qid), gd)
        oks.append(bool(g["ok"]))
        rouges.append(g["rouge"])
        by_calls[_bucket(k)].append(g["rouge"])
        if g["ok"]:
            good_q.add(qid)
    out.update(
        completed=_rate(done), tool_call_rate=_rate([k > 0 for k in calls]), mean_calls=mean(calls),
        calls_hist=dict(sorted(Counter(calls).items())), seen_gold_initial=_rate(seen), good_rate=_rate(oks),
        mean_rouge=mean(rouges) if rouges else 0.0,
        rouge_by_calls={b: mean(v) for b, v in sorted(by_calls.items())},
        empty_rate=_rate(empty), loop_rate=_rate(loops), cap_rate=_rate(caps), mean_tokens=mean(tokens),
        coverage=len(good_q) / len({r["qid"] for r in rows}),
    )
    return out


def _by_group(rows, group_of):
    g = defaultdict(list)
    for r in rows:
        g[group_of(r["qid"])].append(r)
    return g


def summarize(rows, reference_of, gold_dieu_of, grade, group_of=None) -> dict:
    """Overall summary; with group_of(qid) also 'groups': {label: summary}. reference_of / gold_dieu_of: callables qid -> value."""
    ref = reference_of if callable(reference_of) else reference_of.__getitem__
    gd = gold_dieu_of if callable(gold_dieu_of) else (lambda q, d=gold_dieu_of: d.get(q))
    s = _summary(rows, ref, gd, grade)
    if group_of:
        s["groups"] = {k: _summary(v, ref, gd, grade) for k, v in sorted(_by_group(rows, group_of).items(), key=lambda kv: str(kv[0]))}
    return s


def coverage(rows, reference_of, gold_dieu_of, grade, group_of=None) -> dict:
    """Share of questions with at least one good trajectory: {'all': x, group: y}."""
    s = summarize(rows, reference_of, gold_dieu_of, grade, group_of)
    out = {"all": s.get("coverage", 0.0)}
    out.update({k: v.get("coverage", 0.0) for k, v in s.get("groups", {}).items()})
    return out


def should_stop(summary: dict, expects_tools: bool = True) -> list[str]:
    """Reasons to stop and fix the prompt / parser (empty list = ok). Thresholds from Stage 1b."""
    if not summary.get("n"):
        return []
    why = []
    if summary["completed"] < 0.90:
        why.append(f"completed {summary['completed']:.2f} < 0.90")
    if expects_tools and summary["tool_call_rate"] < 0.80:
        why.append(f"tool-call rate {summary['tool_call_rate']:.2f} < 0.80")
    if summary["empty_rate"] > 0.05:
        why.append(f"empty answers {summary['empty_rate']:.2f} > 0.05")
    if summary["loop_rate"] > 0.05:
        why.append(f"loops {summary['loop_rate']:.2f} > 0.05")
    for k, g in summary.get("groups", {}).items():
        if g["n"] >= 30 and g["coverage"] < 0.40:
            why.append(f"coverage {g['coverage']:.2f} < 0.40 in group {k}")
    return why
