import json
import random

import pytest

from corpus.decontam import grams
from train import synth_gen as sg

ART = ("Điều 5. Quy định về mức phạt tiền\n1. Tổ chức vi phạm bị phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng "
       "đối với hành vi không niêm yết công khai biểu phí dịch vụ tại địa điểm giao dịch.\n"
       "2. Cá nhân vi phạm bị phạt bằng một nửa mức phạt đối với tổ chức theo quy định tại khoản 1 Điều này. ") * 2


def art(law="1/2020/TT-A", dieu="5", text=ART):
    return sg.Article(law=law, dieu=dieu, text=text, stem=law)


def test_split_articles():
    text = "Lời nói đầu\n\nĐiều 1. Phạm vi\nNội dung một.\n\nĐiều 2a. Đối tượng\nNội dung hai."
    got = sg.split_articles(text)
    assert [d for d, _ in got] == ["1", "2a"]
    assert got[0][1].startswith("Điều 1.")


def test_pool_filters_exclusion_length_cap(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    big = "x " * 4000
    body = "Nội dung hợp lệ về mức phạt. " * 12
    txt = "\n\n".join([f"Điều 1. A\n{body}", f"Điều 2. B\n{body}", "Điều 3. C\nngắn", f"Điều 4. D\n{big}",
                       f"Điều 5. E\n{body}", f"Điều 6. F\n{body}"])
    (d / "10_2020_TT-X.txt").write_text(txt)
    pool = sg.build_pool([d], excluded={"10/2020/TT-X#2"}, per_law=2, seed=1)
    keys = {a.key for a in pool}
    assert len(pool) == 2 and "10/2020/TT-X#2" not in keys and "10/2020/TT-X#3" not in keys
    assert "10/2020/TT-X#4" not in keys
    assert pool == sg.build_pool([d], excluded={"10/2020/TT-X#2"}, per_law=2, seed=1)  # deterministic
    assert sg.build_pool([tmp_path / "missing"], set(), 5, 1) == []


def test_mix_counts_sum_and_default():
    c = sg.mix_counts(100, sg.DEFAULT_MIX)
    assert c == dict(single=40, multi_same_law=20, multi_law=10, version_family=10, vague=10, no_answer=10)
    assert sum(sg.mix_counts(37, sg.DEFAULT_MIX).values()) == 37


def pool(n_laws=4, per=4):
    out = []
    for i in range(n_laws):
        for j in range(1, per + 1):
            out.append(art(law=f"{i}/2020/TT-L{i}", dieu=str(j), text=ART))
    return out


def test_plan_patterns_and_batch_size():
    batches = sg.plan(pool(), n_items=20, mix=sg.DEFAULT_MIX, seed=3, per_call=10)
    pats = {b.pattern for b in batches}
    assert "version_family" not in pats  # no family_of injected
    assert all(len(b.items) <= 10 for b in batches)
    for b in batches:
        for it in b.items:
            laws = {a.law for a in it.articles}
            if b.pattern == "single" or b.pattern in ("vague", "no_answer"):
                assert len(it.articles) == 1
            if b.pattern == "multi_same_law":
                assert len(laws) == 1 and 2 <= len(it.articles) <= 3
            if b.pattern == "multi_law":
                assert len(laws) == 2
    ids = [b.id for b in batches]
    assert len(ids) == len(set(ids))
    assert ids == [b.id for b in sg.plan(pool(), 20, sg.DEFAULT_MIX, 3, 10)]


def test_version_family_needs_same_number_and_topic():
    a = art("1/2020/TT-A", "5", "Điều 5. Mức phạt tiền vi phạm\n" + "nội dung " * 40)
    b = art("2/2021/TT-A", "5", "Điều 5. Mức phạt tiền vi phạm\n" + "nội dung khác " * 40)
    c = art("3/2021/TT-A", "7", "Điều 7. Mức phạt tiền vi phạm\n" + "nội dung " * 40)
    fam = lambda k: {"1/2020/TT-A", "2/2021/TT-A", "3/2021/TT-A"} - {k}
    mix = dict(version_family=100)
    batches = sg.plan([a, b, c], 5, mix, 1, 10, family_of=fam)
    its = [it for bt in batches for it in bt.items]
    assert its and all({x.dieu for x in it.articles} == {"5"} and len({x.law for x in it.articles}) == 2 for it in its)


def test_prompt_has_sources_and_template_no_test_text():
    b = sg.plan(pool(), 10, dict(single=100), 1, 10)[0]
    p = sg.make_prompt(b)
    assert "JSON" in p and b.items[0].iid in p and "Căn cứ theo Điều" in p
    assert b.items[0].articles[0].text[:50] in p
    nb = sg.plan(pool(), 10, dict(no_answer=100), 1, 10)[0]
    assert "KHÔNG" in sg.make_prompt(nb)


def good_answer(a):
    return f"Căn cứ theo Điều {a.dieu} Thông tư {a.law} quy định về mức phạt tiền như sau:\n" + a.text.split("\n", 1)[1]


Q = "Theo quy định hiện hành, tổ chức không niêm yết công khai biểu phí dịch vụ thì bị xử phạt như thế nào?"


def item_for(a):
    return sg.Item(iid="i1", articles=(a,))


def test_verify_ok_and_reasons():
    a = art()
    it = item_for(a)
    ok = sg.verify({"id": "i1", "question": Q, "answer": good_answer(a)}, it, "single", set(), set())
    assert ok is None
    assert sg.verify({"id": "i1", "question": "ngắn", "answer": good_answer(a)}, it, "single", set(), set()) == "question_len"
    assert sg.verify({"id": "i1", "question": Q, "answer": "ngắn"}, it, "single", set(), set()) == "answer_len"
    fake = "Căn cứ theo Điều 5 Thông tư 1/2020/TT-A quy định về mức phạt như sau:\n" + "Mức phạt là một tỷ đồng cho mọi hành vi vi phạm hành chính bất kỳ trong lĩnh vực này. " * 3
    assert sg.verify({"id": "i1", "question": Q, "answer": fake}, it, "single", set(), set()) == "quote_not_verbatim"
    assert sg.verify({"id": "i1", "question": Q, "answer": good_answer(a)}, it, "single", set(), {"x"}.union(
        grams(Q))) == "near_dup_eval"


def test_verify_whitespace_normalised():
    a = art()
    ans = good_answer(a).replace(" ", "  ").replace("\n", "\n\n")
    assert sg.verify({"id": "i1", "question": Q, "answer": ans}, item_for(a), "single", set(), set()) is None


def test_no_answer_needs_refusal():
    a = art()
    it = item_for(a)
    ref = "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan để trả lời câu hỏi này, vì điều luật cung cấp không đề cập nội dung đó."
    q = "Theo quy định hiện hành, thời hạn cấp giấy phép xây dựng nhà ở riêng lẻ tối đa là bao nhiêu ngày làm việc?"
    assert sg.verify({"id": "i1", "question": q, "answer": ref}, it, "no_answer", set(), set()) is None
    assert sg.verify({"id": "i1", "question": q, "answer": good_answer(a)}, it, "no_answer", set(), set()) == "not_refusal"


def test_parse_and_process_dedup_and_rows():
    a = art()
    b = sg.Batch(id="single-1-0", pattern="single", items=(item_for(a),))
    raw = "```json\n" + json.dumps([{"id": "i1", "question": Q, "answer": good_answer(a)},
                                    {"id": "i1", "question": Q, "answer": good_answer(a)},
                                    {"id": "zz", "question": Q, "answer": "x"}], ensure_ascii=False) + "\n```"
    rows, rej = sg.process(raw, b, set(), set(), seen=set())
    assert len(rows) == 1 and rows[0]["pattern"] == "single" and rows[0]["ok"] is True
    assert rows[0]["sources"] == ["1/2020/TT-A#5"]
    reasons = sorted(r["reason"] for r in rej)
    assert reasons == ["duplicate", "unknown_id"]
    rows, rej = sg.process("không phải json", b, set(), set(), seen=set())
    assert rows == [] and rej[0]["reason"] == "bad_json"


def test_estimate_and_budget():
    b = sg.plan(pool(), 10, dict(single=100), 1, 10)[0]
    assert sg.estimate_tokens(sg.make_prompt(b), len(b.items)) > len(sg.make_prompt(b)) / 3.5
    assert sg.within_budget(used=10, add=5, max_tokens=20) and not sg.within_budget(10, 11, 20)


def test_resume_done_ids(tmp_path):
    p = tmp_path / "done.txt"
    assert sg.read_done(p) == set()
    sg.mark_done(p, "a")
    sg.mark_done(p, "b")
    assert sg.read_done(p) == {"a", "b"}


def test_pacer_delay():
    assert sg.min_interval(20) == pytest.approx(3.0)


def test_pool_focus_laws_get_more_articles_and_other_laws_are_thinned(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    body = "Nội dung hợp lệ về mức phạt. " * 12
    txt = "\n\n".join(f"Điều {i}. T{i}\n{body}" for i in range(1, 9))
    for name in ("10_2020_TT-FOCUS", "11_2020_TT-A", "12_2020_TT-B", "13_2020_TT-C", "14_2020_TT-D", "15_2020_TT-E"):
        (d / f"{name}.txt").write_text(txt)
    pool = sg.build_pool([d], set(), per_law=2, seed=1, focus={"10/2020/TT-FOCUS"}, per_law_focus=6, other_frac=0.4)
    by = {}
    for a in pool:
        by.setdefault(a.law, 0)
        by[a.law] += 1
    assert by["10/2020/TT-FOCUS"] == 6
    others = [k for k in by if k != "10/2020/TT-FOCUS"]
    assert 0 < len(others) < 5 and all(by[k] == 2 for k in others)
    assert pool == sg.build_pool([d], set(), per_law=2, seed=1, focus={"10/2020/TT-FOCUS"}, per_law_focus=6, other_frac=0.4)


def test_pool_without_focus_is_unchanged(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    body = "Nội dung hợp lệ về mức phạt. " * 12
    (d / "10_2020_TT-X.txt").write_text("\n\n".join(f"Điều {i}. T\n{body}" for i in range(1, 6)))
    assert len(sg.build_pool([d], set(), per_law=3, seed=1)) == 3
