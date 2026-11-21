import json

from train.react_data import grade
from train.traj_stats import coverage, should_stop, summarize

REF = "Theo Điều 5 Thông tư 50/2024/TT-NHNN, mức lãi suất là 1%."
GOOD = REF
BAD = "Không rõ."


def call(cid, name="get_article", **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def row(qid, answer=GOOD, calls=(), prompt="Điều 5 nói về lãi suất"):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": prompt}]
    for i, (name, args) in enumerate(calls):
        msgs += [call(f"c{i}", name, **args), {"role": "tool", "tool_call_id": f"c{i}", "content": "Điều 5. Lãi suất"}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return {"qid": qid, "question": "q", "messages": msgs}


def S(rows, **kw):
    return summarize(rows, {r["qid"]: REF for r in rows}, {r["qid"]: {"5"} for r in rows}, grade, **kw)


def test_basic_rates():
    s = S([row("a"), row("b", BAD, [("get_article", {"n": 1})]), row("c", None, [("get_article", {"n": 1})])])
    assert s["n"] == 3
    assert abs(s["completed"] - 2 / 3) < 1e-9
    assert abs(s["tool_call_rate"] - 2 / 3) < 1e-9
    assert abs(s["mean_calls"] - 2 / 3) < 1e-9
    assert s["calls_hist"] == {0: 1, 1: 2}
    assert abs(s["good_rate"] - 1 / 3) < 1e-9
    assert abs(s["empty_rate"] - 0) < 1e-9


def test_empty_answer_and_not_completed():
    s = S([row("a", "  ")])
    assert s["empty_rate"] == 1 and s["completed"] == 0


def test_loop_and_cap():
    same = [("get_article", {"n": 1})] * 3
    s = S([row("a", None, same), row("b", GOOD, [("get_article", {"n": 1}), ("get_article", {"n": 2})])])
    assert s["loop_rate"] == 0.5
    assert s["cap_rate"] == 0.5  # >=3 calls, no final answer


def test_seen_gold_in_initial_prompt():
    s = S([row("a"), row("b", prompt="nothing here")])
    assert s["seen_gold_initial"] == 0.5


def test_rouge_by_calls_buckets():
    rows = [row("a"), row("b", GOOD, [("t", {"i": i}) for i in range(4)])]
    s = S(rows)
    assert set(s["rouge_by_calls"]) == {"0", "3+"}
    assert s["rouge_by_calls"]["0"] > 0.9


def test_tokens_estimate():
    s = S([row("a", "x" * 350, prompt="y" * 350)])
    assert s["mean_tokens"] >= 200


def test_groups_and_coverage():
    rows = [row("a"), row("b", BAD), row("b", GOOD), row("c", BAD)]
    g = {"a": "train", "b": "train", "c": "dev"}.get
    s = S(rows, group_of=g)
    assert s["groups"]["train"]["n"] == 3 and s["groups"]["dev"]["n"] == 1
    cov = coverage(rows, {q: REF for q in "abc"}, {q: {"5"} for q in "abc"}, grade, group_of=g)
    assert cov == {"all": 2 / 3, "train": 1.0, "dev": 0.0}
    assert s["coverage"] == 2 / 3


def test_should_stop_ok_and_reasons():
    s = S([row("a")] * 5, group_of=lambda q: "g")
    assert should_stop(s, expects_tools=False) == []
    assert any("tool" in r for r in should_stop(s, expects_tools=True))
    bad = S([row("a", None)] * 5)
    reasons = should_stop(bad, expects_tools=False)
    assert any("completed" in r for r in reasons)


def test_should_stop_group_coverage_needs_n30():
    rows = [row(f"q{i}", BAD) for i in range(30)]
    s = S(rows, group_of=lambda q: "g")
    assert any("coverage" in r for r in should_stop(s, expects_tools=False))
    s = S(rows[:10], group_of=lambda q: "g")
    assert not any("coverage" in r for r in should_stop(s, expects_tools=False))
