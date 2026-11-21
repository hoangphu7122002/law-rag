import random

from generate.qprofile import question_profile
from generate.rprofile import retrieval_profile
from kg import KG
from records import Passage


def P(doc, dieu, title="Quy định về cho vay", body="nội dung", score=1.0):
    return Passage(pid=f"{doc}#{dieu}", doc=doc, text=f"{title}\nĐiều {dieu}. Tiêu đề điều {dieu}\n{body}", score=score)


def make_kg():
    kg = KG(":memory:")
    kg.upsert_doc("a", key="39/2016/TT-NHNN", title="Thông tư 39/2016", status="Còn hiệu lực một phần", eff_from="2017-03-15")
    kg.upsert_doc("b", key="06/2023/TT-NHNN", title="Thông tư 06/2023", status="Còn hiệu lực", eff_from="2023-09-01")
    kg.add_edge("b", "a", "amends", "sửa đổi")
    kg.commit()
    return kg


def test_lists_top_laws_in_rank_order_with_dieu_and_status():
    ps = [P("06_2023_TT-NHNN", "1"), P("39_2016_TT-NHNN", "8"), P("39_2016_TT-NHNN", "9")]
    text, sig = retrieval_profile(question_profile("Cho vay là gì?"), ps, make_kg())
    assert text.index("06/2023/TT-NHNN") < text.index("39/2016/TT-NHNN")
    assert "Điều 8" in text and "Điều 9" in text and "Còn hiệu lực một phần" in text


def test_named_law_missing_from_the_result_is_a_missing_signal():
    ps = [P("06_2023_TT-NHNN", "1")]
    _, sig = retrieval_profile(question_profile("Theo Thông tư 10/2025/TT-NHNN thì sao?"), ps, make_kg())
    assert {"kind": "law_missing", "law": "10/2025/TT-NHNN"} in sig


def test_named_law_present_has_no_missing_signal():
    ps = [P("10_2025_TT-NHNN", "3")]
    _, sig = retrieval_profile(question_profile("Theo Thông tư 10/2025/TT-NHNN thì sao?"), ps, make_kg())
    assert not [s for s in sig if s["kind"] == "law_missing"]


def test_named_article_not_retrieved_is_a_missing_signal():
    ps = [P("23_2014_TT-NHNN", "4")]
    _, sig = retrieval_profile(question_profile("Điều 5 Thông tư 23/2014/TT-NHNN quy định gì?"), ps, make_kg())
    assert {"kind": "dieu_missing", "law": "23/2014/TT-NHNN", "dieu": "5"} in sig


def test_partial_number_matches_a_retrieved_law_prefix():
    ps = [P("39_2016_TT-NHNN", "8")]
    _, sig = retrieval_profile(question_profile("Dự thảo sửa Thông tư 39/2016 có gì?"), ps, make_kg())
    assert not [s for s in sig if s["kind"] == "law_missing"]


def test_appendix_flag_without_an_appendix_passage_is_a_missing_kind():
    ps = [P("10_2025_TT-NHNN", "3")]
    _, sig = retrieval_profile(question_profile("Mẫu quyết định kèm phụ lục của Thông tư 10/2025/TT-NHNN?"), ps, make_kg())
    assert {"kind": "flag_missing", "flag": "phu_luc"} in sig


def test_same_family_laws_are_reported_neutrally():
    ps = [P("06_2023_TT-NHNN", "1"), P("39_2016_TT-NHNN", "8")]
    text, sig = retrieval_profile(question_profile("Cho vay là gì?"), ps, make_kg())
    assert {"kind": "same_family", "laws": ["06/2023/TT-NHNN", "39/2016/TT-NHNN"]} in sig
    assert "không ưu tiên" in text and "mới hơn" not in text


def test_shuffled_listing_changes_the_order_but_not_the_signals():
    ps = [P(f"{i}_2020_TT-NHNN", "1") for i in range(10, 16)]
    q = question_profile("Cho vay là gì?")
    t0, s0 = retrieval_profile(q, ps, make_kg())
    t1, s1 = retrieval_profile(q, ps, make_kg(), rng=random.Random(1), shuffle=1.0)
    assert s0 == s1 and t0 != t1


def test_profile_stays_short_and_has_no_article_body():
    ps = [P(f"{i}_2020_TT-NHNN", str(j), body="X" * 3000) for i in range(10, 15) for j in range(1, 6)]
    text, _ = retrieval_profile(question_profile("Cho vay là gì?"), ps, make_kg())
    assert len(text) < 2200 and "XXXX" not in text
