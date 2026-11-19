import json

from train.react_build import select_s1
from train.react_data import grade

REF = "Theo Điều 5 Thông tư 50/2024/TT-NHNN, mức lãi suất là 1%."
GOOD = "Theo Điều 5 Thông tư 50/2024/TT-NHNN, mức lãi suất là 1%."


def call(cid, name="get_article", **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def row(qid, answer=GOOD, tools=()):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "câu hỏi?"}]
    for i, name in enumerate(tools):
        msgs += [call(f"c{i}", name), {"role": "tool", "tool_call_id": f"c{i}", "content": "Điều 5. Lãi suất"}]
    return {"qid": qid, "question": "câu hỏi?", "messages": msgs + [{"role": "assistant", "content": answer}]}


def select(rows, keep=("q1",), **kw):
    return select_s1(rows, {"q1": REF, "q2": REF}, {"q1": {"5"}, "q2": {"5"}}, set(keep), grade, **kw)


def test_keeps_a_good_trajectory_of_a_kept_question():
    r = row("q1")
    assert select([r]) == [r]


def test_drops_questions_outside_keep_qids_and_wrong_answers():
    rows = [row("q2"), row("q1", answer="Theo Điều 9, mức lãi suất là 1%.")]
    assert select(rows) == []


def test_drops_a_trajectory_without_final_answer_or_with_a_dangling_tool_call():
    empty = row("q1", answer="")
    dangling = row("q1", tools=["get_article"])
    dangling["messages"][-1] = call("c9")      # last turn is a tool call, never answered
    no_result = row("q1", tools=["get_article"])
    del no_result["messages"][-2]              # tool call whose result is missing
    assert select([empty, dangling, no_result]) == []


def test_keeps_only_the_best_per_question_ties_go_to_fewer_tool_calls():
    worse = row("q1", answer=GOOD + " Ngoài ra còn nhiều quy định khác không liên quan.")
    many, few = row("q1", tools=["get_article", "search_law"]), row("q1")
    assert select([worse, many, few]) == [few]
    assert select([worse, many, few], max_per_question=2) == [few, many]


def test_stats_counts_kept_coverage_and_tool_use():
    from train.react_build import stats

    rows = [row("q1", tools=["get_article"]), row("q2"), row("q3")]
    allr = rows + [row("q1"), row("q2")]
    s = stats(rows[:2], allr)
    assert s["kept"] == 2 and s["questions_kept"] == 2 and s["questions_total"] == 3
    assert s["coverage"] == 2 / 3
    assert s["share_with_tool"] == 0.5 and s["share_with_law_tool"] == 0.5
    assert stats([], [])["share_with_tool"] == 0.0


def test_gold_dieu_takes_the_article_number_of_each_gold_pair():
    from train.react_build import gold_dieu

    assert gold_dieu({"50/2024/TT-NHNN#5", "01/2020/QH14#12a"}) == {"5", "12a"}
    assert gold_dieu(None) == set()
