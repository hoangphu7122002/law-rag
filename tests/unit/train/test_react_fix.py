import json

from train.react_fix import parse_repair

GOOD = {"actions": [{"tool": "get_article", "law": "50/2024/TT-NHNN", "dieu": 5}], "answer": "Căn cứ Điều 5 quy định X."}


def test_parse_plain_and_fenced_and_prose():
    assert parse_repair(json.dumps(GOOD)) == GOOD
    assert parse_repair("```json\n" + json.dumps(GOOD) + "\n```") == GOOD
    assert parse_repair("Đây là kết quả:\n" + json.dumps(GOOD) + "\nHết.") == GOOD


def test_parse_rejects_invalid():
    bad = [
        "not json",
        json.dumps({"actions": [{"tool": "rm_rf"}], "answer": "x"}),
        json.dumps({"actions": [{"tool": "search_law"}], "answer": "x"}),
        json.dumps({"actions": [{"tool": "get_article", "law": "a", "dieu": "x"}], "answer": "x"}),
        json.dumps({"actions": [{"tool": "search_law", "query": "q"}] * 3, "answer": "x"}),
        json.dumps({"actions": [], "answer": ""}),
        json.dumps({"actions": []}),
    ]
    assert all(parse_repair(t) is None for t in bad)


def test_parse_batch_by_id_skips_invalid_items():
    from train.react_fix import parse_batch

    text = "Kết quả:\n```json\n" + json.dumps([{"id": "q1", **GOOD}, {"id": "q2", "actions": [{"tool": "x"}], "answer": "a"}, {"id": 3, **GOOD}]) + "\n```"
    out = parse_batch(text)
    assert out == {"q1": GOOD, "3": GOOD}
    assert parse_batch("garbage") == {}


REF = "Căn cứ Điều 5 Thông tư 50/2024/TT-NHNN, ngân hàng phải lưu trữ hồ sơ giao dịch trong thời hạn mười năm kể từ ngày kết thúc giao dịch để phục vụ kiểm tra giám sát."
ARTICLE = "Điều 5. Lưu trữ hồ sơ\nNgân hàng lưu trữ hồ sơ giao dịch."


def fake_tool(name, args):
    return ARTICLE if name == "get_article" else "[1] (Luật khác)\nĐiều 9. Nội dung khác"


def fake_grade(answer, reference, gold_dieu):
    return {"ok": "Điều 5" in answer}


def verify(repair, **kw):
    from train.react_fix import verify_repair

    return verify_repair(repair, REF, {"5"}, kw.pop("run_tool", fake_tool), kw.pop("grade", fake_grade), **kw)


def test_verify_accepts_good_repair():
    good = {"actions": GOOD["actions"], "answer": "Theo Điều 5, hồ sơ giao dịch được ngân hàng lưu giữ trong mười năm."}
    assert verify(good) == {"ok": True, "reason": "ok"}


def test_verify_rejects_each_failure():
    ans = "Theo Điều 5, hồ sơ giao dịch được ngân hàng lưu giữ trong mười năm."
    assert verify(None)["reason"] == "invalid"
    assert verify({"actions": [{"tool": "x"}], "answer": ans})["reason"] == "invalid"
    miss = {"actions": [{"tool": "search_law", "query": "lưu trữ"}], "answer": ans}
    assert verify(miss) == {"ok": False, "reason": "gold_not_retrieved"}
    bad_ans = {"actions": GOOD["actions"], "answer": "Hồ sơ được lưu mười năm."}
    assert verify(bad_ans)["reason"] == "answer_not_ok"
    copy = {"actions": GOOD["actions"], "answer": REF}
    assert verify(copy)["reason"] == "copies_reference"


def test_verify_gold_must_be_whole_number_heading():
    # Điều 50 must not count as Điều 5
    r = {"actions": GOOD["actions"], "answer": "Theo Điều 5, hồ sơ giao dịch được ngân hàng lưu giữ trong mười năm."}
    assert verify(r, run_tool=lambda n, a: "Điều 50. Khác")["reason"] == "gold_not_retrieved"


def test_copy_share_threshold_and_short_answers():
    from train.react_fix import copy_share

    assert copy_share("ngắn quá", REF) == 0.0
    assert copy_share(REF, REF) == 1.0
    half = REF + " thêm một đoạn khác hoàn toàn mới lạ không có trong tham chiếu gì cả nhé các bạn ơi " + REF.upper()[:0]
    assert 0.0 < copy_share(half, REF) < 1.0


PREFIX = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Hồ sơ giao dịch lưu bao lâu?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "s1", "type": "function", "function": {"name": "search_law", "arguments": json.dumps({"query": "sai"})}}]},
    {"role": "tool", "tool_call_id": "s1", "content": "[1] Điều 9. " + "x" * 5000},
    {"role": "assistant", "content": "Không rõ."},
]


def test_build_repair_request_contents():
    from train.react_fix import build_repair_request

    p = build_repair_request(PREFIX, REF, {"5"}, "50/2024/TT-NHNN", bad_step=2)
    assert "Hồ sơ giao dịch lưu bao lâu?" in p and REF in p and "50/2024/TT-NHNN" in p and "Điều 5" in p
    assert "get_article" in p and "search_law" in p and "JSON" in p
    assert "Không rõ." not in p            # only messages[:bad_step]
    assert len(p) < 4000                   # compact: long tool output truncated


def test_build_batch_prompt_numbers_items():
    from train.react_fix import build_batch_prompt

    p = build_batch_prompt(["AAA", "BBB"])
    assert "AAA" in p and "BBB" in p and '"id"' in p
    assert p.index("AAA") < p.index("BBB")


def test_to_messages_roundtrip_to_rows():
    from train.react_data import to_turn_rows
    from train.react_fix import to_messages

    repair = {"actions": GOOD["actions"], "answer": GOOD["answer"]}
    out = to_messages(PREFIX[:2], repair, [ARTICLE])
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    call = out[2]["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "get_article"
    assert json.loads(call["function"]["arguments"]) == {"law": "50/2024/TT-NHNN", "dieu": 5}
    assert out[3] == {"role": "tool", "tool_call_id": call["id"], "content": ARTICLE}
    assert out[4] == {"role": "assistant", "content": GOOD["answer"]}
    rows = to_turn_rows(out)
    assert len(rows) == 2 and rows[1]["prompt"] == out[:4]


def test_to_messages_no_actions_and_unique_ids():
    from train.react_fix import to_messages

    two = {"actions": [{"tool": "search_law", "query": "a"}, {"tool": "search_law", "query": "b"}], "answer": "ok"}
    out = to_messages(PREFIX[:2], two, ["r1", "r2"])
    ids = [c["id"] for m in out if "tool_calls" in m for c in m["tool_calls"]]
    assert len(set(ids)) == 2 and [m["role"] for m in out].count("tool") == 2
    assert to_messages(PREFIX[:2], {"actions": [], "answer": "ok"}, []) == PREFIX[:2] + [{"role": "assistant", "content": "ok"}]
