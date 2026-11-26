import json

from train.cpt_data import articles
from train.cpt_synth import Article, batches, make_prompt, process, verify_text

ART = Article(key="49/2005/QH11#13", title="Luật Các công cụ chuyển nhượng số 49/2005/QH11",
              text="Điều 13. Mất công cụ chuyển nhượng\n1. Khi công cụ chuyển nhượng bị mất, người thụ hưởng phải thông báo ngay bằng văn bản.\n2. Thời hạn là 30 ngày kể từ ngày mất.")


def test_articles_merges_the_chunks_of_one_article_in_order_and_drops_the_repeated_title_line():
    nodes = [("a", "7", 2, "Luật A số 1/2000/QH\nphần hai của điều"), ("a", "7", 1, "Luật A số 1/2000/QH\nĐiều 7. Tiêu đề\nphần một"), ("b", "1", 1, "Luật B\nĐiều 1. x"),
             ("a", None, 0, "Luật A số 1/2000/QH\nlời nói đầu")]
    out = articles(nodes, wanted={"a"})
    assert out == [{"key": "a#7", "title": "Luật A số 1/2000/QH", "text": "Điều 7. Tiêu đề\nphần một\nphần hai của điều"}]


def test_batches_group_articles_and_the_prompt_names_every_article_and_the_three_output_kinds():
    bs = batches([ART, ART, ART], per_call=2)
    assert [len(b.items) for b in bs] == [2, 1]
    p = make_prompt(bs[0])
    assert "49/2005/QH11#13" in p and "paraphrases" in p and "qa_fwd" in p and "qa_rev" in p


def test_verify_text_rejects_new_numbers_a_missing_article_number_and_near_copies():
    ok = "Theo Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11, khi công cụ bị mất thì người thụ hưởng phải báo bằng văn bản, thời hạn là 30 ngày."
    assert verify_text(ok, ART, "paraphrase") is None
    assert verify_text(ok.replace("30 ngày", "45 ngày"), ART, "paraphrase") == "new_numbers"
    assert verify_text(ok.replace("Điều 13", "Điều 14"), ART, "paraphrase") in ("wrong_article", "new_numbers")
    assert verify_text("Điều 13 " + ART.text, ART, "paraphrase") == "near_copy"


def test_process_turns_a_good_reply_into_rows_and_a_bad_one_into_a_reject():
    raw = json.dumps([{"id": "0", "paraphrases": ["Theo Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11, người thụ hưởng khi bị mất công cụ phải thông báo bằng văn bản, thời hạn 30 ngày."],
                       "qa_fwd": [{"q": "Khi công cụ chuyển nhượng bị mất thì làm gì?", "a": "Căn cứ theo Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11, người thụ hưởng phải thông báo ngay bằng văn bản."}],
                       "qa_rev": [{"q": "Quy định nào nói về thời hạn 30 ngày kể từ ngày công cụ bị mất?", "a": "Quy định này nằm tại Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11."}]}])
    rows, rej = process(raw, batches([ART], 1)[0], blocked=set())
    assert {r["source"] for r in rows} == {"paraphrase", "qa_fwd", "qa_rev"} and not rej
    assert all(r["key"] == ART.key for r in rows)
    rows, rej = process("không phải json", batches([ART], 1)[0], blocked=set())
    assert rows == [] and rej[0]["reason"] == "bad_json"


def test_summaries_are_a_fourth_kind_that_must_be_shorter_than_the_article_and_keep_its_numbers():
    s = "Tóm tắt Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11: công cụ bị mất thì phải báo văn bản, thời hạn 30 ngày."
    assert "summaries" in make_prompt(batches([ART], 1)[0])
    assert verify_text(s, ART, "summary") is None
    assert verify_text(s + " " + ART.text * 2, ART, "summary") == "too_long"
    raw = json.dumps([{"id": "0", "summaries": [s]}])
    rows, rej = process(raw, batches([ART], 1)[0], blocked=set())
    assert [r["source"] for r in rows] == ["summary"] and not rej


def test_process_accepts_ids_the_model_decorates_such_as_id_equals_0():
    s = "Tóm tắt Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11: công cụ bị mất thì phải báo văn bản, thời hạn 30 ngày."
    for rid in ("id=0", 0, "0", " 0 "):
        rows, rej = process(json.dumps([{"id": rid, "summaries": [s]}]), batches([ART], 1)[0], blocked=set())
        assert len(rows) == 1 and not rej


def test_process_matches_an_item_by_its_key_when_the_model_echoes_the_key_as_id():
    s = "Tóm tắt Điều 13 Luật Các công cụ chuyển nhượng số 49/2005/QH11: công cụ bị mất thì phải báo văn bản, thời hạn 30 ngày."
    for rid in ("49/2005/QH11#13", "0 | 49/2005/QH11#13"):
        rows, rej = process(json.dumps([{"id": rid, "summaries": [s]}]), batches([ART], 1)[0], blocked=set())
        assert len(rows) == 1 and not rej


def _arts():
    mk = lambda n, t: Article(key=f"49/2005/QH11#{n}", title="Luật Các công cụ chuyển nhượng số 49/2005/QH11", text=f"Điều {n}. Tiêu đề\n{t}")   # noqa: E731
    return [mk(13, "Khi công cụ bị mất, người thụ hưởng thông báo ngay. Thời hạn là 30 ngày."), mk(14, "Việc thông báo thực hiện bằng văn bản gửi người ký phát trong 5 ngày."),
            mk(15, "Người ký phát phải trả tiền trong 10 ngày kể từ khi nhận thông báo.")]


def test_groups_join_consecutive_articles_of_the_same_law_only():
    from train.cpt_synth import groups
    other = Article(key="77/2010/QH12#1", title="Luật Khác", text="Điều 1. x\nnội dung")
    gs = groups([*_arts(), other], size=2)
    assert [g.key for g in gs] == ["49/2005/QH11#13-14", "49/2005/QH11#15"] or [g.key for g in gs][0] == "49/2005/QH11#13-14"
    assert all(len({a.law for a in g.items}) == 1 for g in gs)


def test_multi_prompt_and_process_keep_valid_multi_article_scenario_and_summary_rows():
    from train.cpt_synth import groups, make_prompt_multi, process_multi
    g = groups(_arts(), size=3)
    b = batches(g, 1)[0]
    assert "multi_qa" in make_prompt_multi(b) and "scenario" in make_prompt_multi(b) and "49/2005/QH11#13-15" in make_prompt_multi(b)
    raw = json.dumps([{"id": "0",
                       "multi_qa": [{"q": "Người thụ hưởng phải làm gì và trong bao lâu khi mất công cụ, và người ký phát phải trả tiền sau bao lâu?",
                                     "a": "Theo Điều 13 và Điều 15 Luật Các công cụ chuyển nhượng số 49/2005/QH11, người thụ hưởng thông báo ngay trong 30 ngày, người ký phát trả tiền trong 10 ngày."}],
                       "scenario": [{"q": "Ông A bị mất công cụ chuyển nhượng, ông cần thông báo cho ai?",
                                     "a": "Theo Điều 14 Luật Các công cụ chuyển nhượng số 49/2005/QH11, ông A thông báo bằng văn bản gửi người ký phát trong 5 ngày."}],
                       "summary": ["Tóm tắt Điều 13, 14, 15 Luật Các công cụ chuyển nhượng số 49/2005/QH11: mất công cụ thì thông báo trong 30 ngày, bằng văn bản trong 5 ngày, trả tiền trong 10 ngày."]}], ensure_ascii=False)
    rows, rej = process_multi(raw, b, blocked=set())
    assert sorted(r["source"] for r in rows) == ["law_summary", "qa_multi", "scenario"] and not rej
    bad = raw.replace("Điều 14 Luật", "Điều 99 Luật")
    rows, rej = process_multi(bad, b, blocked=set())
    assert "scenario" not in {r["source"] for r in rows} and any(r["reason"] == "wrong_article" for r in rej)
