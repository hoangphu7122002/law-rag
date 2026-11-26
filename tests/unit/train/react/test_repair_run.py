import json

from train.react.repair_run import RULES, estimate_tokens, run_repairs, select_items

REF = "Căn cứ theo Điều 5 Thông tư 50/2024/TT-NHNN, ngân hàng phải lưu trữ hồ sơ giao dịch trong thời hạn mười năm kể từ ngày kết thúc giao dịch như sau: hồ sơ gốc."
ARTICLE = "Điều 5. Lưu trữ hồ sơ\nNgân hàng lưu trữ hồ sơ giao dịch mười năm."
GOOD_ANS = "Căn cứ theo Điều 5 Thông tư 50/2024/TT-NHNN, ngân hàng phải lưu trữ hồ sơ giao dịch trong thời hạn mười năm kể từ ngày kết thúc giao dịch như sau: hồ sơ gốc."
# copy share would be 1.0 for GOOD_ANS == REF; use a paraphrase sharing no 13-gram
PARA = "Căn cứ theo Điều 5 Thông tư 50/2024/TT-NHNN, hồ sơ giao dịch phải được ngân hàng giữ mười năm kể từ khi giao dịch xong như sau: hồ sơ gốc."


def traj(qid, content="Không biết."):
    return {"qid": qid, "question": "q?", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Hỏi " + qid}, {"role": "assistant", "content": content}]}


def tables(qids):
    qa = {q: {"answer": REF} for q in qids}
    gold = {q: {"50/2024/TT-NHNN#5"} for q in qids}
    return qa, gold


def fake_tool(name, args):
    return ARTICLE


def make_teacher(answer=PARA, drop=()):
    calls = []

    def teacher(prompt):
        calls.append(prompt)
        import re
        ids = re.findall(r"=== Mục id=(\d+) ===", prompt)
        out = [{"id": i, "actions": [{"tool": "get_article", "law": "50/2024/TT-NHNN", "dieu": 5}], "answer": answer} for i in ids if i not in drop]
        return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"

    teacher.calls = calls
    return teacher


def go(tmp_path, rows, teacher, **kw):
    qa, gold = tables([r["qid"] for r in rows])
    items = select_items(rows, qa, gold, done=kw.pop("done", set()), limit=kw.pop("limit", None))
    kw.setdefault("sleep", lambda s: None)
    return items, run_repairs(items, teacher, fake_tool, tmp_path / "ok.jsonl", tmp_path / "rej.jsonl", **kw)


def test_select_skips_good_done_and_limits():
    qa, gold = tables(["a", "b", "c"])
    good = traj("a", REF)
    good["messages"].insert(2, {"role": "tool", "content": ARTICLE})  # gold seen -> first_bad_step None
    rows = [good, traj("b"), traj("c")]
    assert [i["qid"] for i in select_items(rows, qa, gold)] == ["b", "c"]
    assert [i["qid"] for i in select_items(rows, qa, gold, done={"b"})] == ["c"]
    assert [i["qid"] for i in select_items(rows, qa, gold, limit=1)] == ["b"]


def test_accepted_written_and_batched(tmp_path):
    rows = [traj(f"q{i}") for i in range(5)]
    t = make_teacher()
    items, stats = go(tmp_path, rows, t, batch_size=2)
    assert len(t.calls) == 3 and stats["accepted"] == 5
    out = [json.loads(l) for l in (tmp_path / "ok.jsonl").read_text().splitlines()]
    assert {o["qid"] for o in out} == {f"q{i}" for i in range(5)}
    assert out[0]["source"] == "repair" and out[0]["messages"][-1]["content"] == PARA and out[0]["grade"]["ok"]
    assert out[0]["messages"][:3] == rows[0]["messages"][:2] + [] or out[0]["messages"][0]["role"] == "system"
    assert "Căn cứ theo Điều" in t.calls[0] and "JSON" in t.calls[0] and RULES in t.calls[0]


def test_rejected_with_reason_and_missing(tmp_path):
    rows = [traj("a"), traj("b")]
    items, stats = go(tmp_path, rows, make_teacher(answer=REF, drop=("1",)))
    rej = {json.loads(l)["qid"]: json.loads(l)["reason"] for l in (tmp_path / "rej.jsonl").read_text().splitlines()}
    assert rej == {"a": "copies_reference", "b": "no_output"} and stats["accepted"] == 0


def test_resume_via_done_and_files_appended(tmp_path):
    rows = [traj("a"), traj("b")]
    go(tmp_path, rows[:1], make_teacher())
    from train.react.repair_run import done_qids
    assert done_qids(tmp_path / "ok.jsonl", tmp_path / "rej.jsonl") == {"a"}
    t = make_teacher()
    go(tmp_path, rows, t, done=done_qids(tmp_path / "ok.jsonl"))
    assert len(t.calls) == 1 and len((tmp_path / "ok.jsonl").read_text().splitlines()) == 2


def test_caps_and_pacing(tmp_path):
    rows = [traj(f"q{i}") for i in range(6)]
    t = make_teacher()
    _, stats = go(tmp_path, rows, t, batch_size=2, max_calls=2)
    assert len(t.calls) == 2 and stats["stopped"] == "max_calls"
    t2 = make_teacher()
    _, stats = go(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, rows, t2, batch_size=2, max_tokens=10)
    assert len(t2.calls) == 0 and stats["stopped"] == "max_tokens"
    # pacing: 2 calls/min -> 30 s between calls
    now = [0.0]
    sleeps = []
    (tmp_path / "p").mkdir()
    go(tmp_path / "p", rows, make_teacher(), batch_size=2, calls_per_min=2, clock=lambda: now[0], sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    assert sleeps == [30.0, 30.0]


def test_estimate_tokens():
    assert estimate_tokens("x" * 350) == 100
