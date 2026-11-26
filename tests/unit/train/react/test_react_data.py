import json

from train.react.react_data import first_bad_step, grade, to_turn_rows

GOLD = {("50/2024/TT-NHNN", "5")}


def msgs(*turns):
    out = [{"role": "system", "content": "sys"}, {"role": "user", "content": "câu hỏi?"}]
    for role, content, *calls in turns:
        m = {"role": role, "content": content}
        if calls:
            m["tool_calls"] = [{"id": "c", "type": "function", "function": {"name": calls[0][0], "arguments": json.dumps(calls[0][1])}}]
        out.append(m)
    return out


def test_grade_scores_the_final_answer_and_checks_the_gold_dieu_is_cited():
    ref = "Theo Điều 5 Thông tư 50/2024/TT-NHNN, mức lãi suất là 1%."
    good = grade("Theo Điều 5 Thông tư 50/2024/TT-NHNN, mức lãi suất là 1%.", ref, {"5"})
    wrong_dieu = grade("Theo Điều 9, mức lãi suất là 1%.", ref, {"5"})
    assert good["ok"] and good["rouge"] > 0.9 and good["cites_gold"]
    assert not wrong_dieu["ok"] and not wrong_dieu["cites_gold"]


def test_first_bad_step_is_the_first_turn_that_does_not_move_towards_the_gold_dieu():
    # the first assistant turn searches, the tool result lacks the gold Điều, the second turn answers from the wrong context
    traj = msgs(("assistant", "", ("search_law", {"query": "lãi suất"})), ("tool", "[1] (a)\nĐiều 9. Khác"), ("assistant", "Theo Điều 9..."))
    assert first_bad_step(traj, GOLD_DIEU := {"5"}) == 2          # index in the message list of the answer that never saw the gold Điều
    seen = msgs(("assistant", "", ("get_article", {"law": "50/2024/TT-NHNN", "dieu": 5})), ("tool", "Điều 5. Lãi suất"), ("assistant", "Theo Điều 5..."))
    assert first_bad_step(seen, GOLD_DIEU) is None                  # the gold Điều reached the model: nothing to fix before the answer


def test_each_assistant_turn_becomes_one_training_row_with_its_prefix():
    traj = msgs(("assistant", "", ("get_article", {"law": "x", "dieu": 5})), ("tool", "Điều 5. Nội dung"), ("assistant", "Đáp án."))
    rows = to_turn_rows(traj)
    assert len(rows) == 2
    assert [m["role"] for m in rows[0]["prompt"]] == ["system", "user"] and rows[0]["completion"]["tool_calls"]
    assert [m["role"] for m in rows[1]["prompt"]] == ["system", "user", "assistant", "tool"] and rows[1]["completion"]["content"] == "Đáp án."
