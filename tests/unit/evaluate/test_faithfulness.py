from evaluate.faithfulness import (citation_match, dieu_numbers, latex_escape, law_ids, law_match, ngram_fidelity, number_grounding,
                                   numbers, pick_by_quantile)

REF = "Căn cứ tại khoản 2 Điều 17 Thông tư 50/2024/TT-NHNN quy định mức phạt 1.000.000 đồng."


def test_articles_and_law_ids_are_extracted():
    assert dieu_numbers("Theo Điều 6 và điều 17 khoản 2") == {6, 17}
    assert law_ids("Nghị định 100/2019/NĐ-CP sửa đổi bởi 123/2021/NĐ-CP") == {"100/2019/nđ-cp", "123/2021/nđ-cp"}


def test_citation_and_law_match_against_the_reference():
    assert citation_match("Căn cứ Điều 17 quy định", REF) is True
    assert citation_match("Căn cứ Điều 9 quy định", REF) is False
    assert citation_match("không có điều nào", "Câu trả lời không trích dẫn") is None      # nothing to check
    assert law_match("theo Thông tư 50/2024/TT-NHNN", REF) is True and law_match("Nghị định 1/2020/NĐ-CP", REF) is False


def test_numbers_are_normalised_and_grounded_in_the_sources_or_question():
    assert numbers("phạt 1.000.000 đồng, 5% và khoản 2") == {"1000000", "5", "2"}
    g = number_grounding("phạt 1.000.000 đồng và 7 ngày", ["Mức phạt 1000000 đồng"], question="")
    assert g == 0.5                                                                        # 1000000 found, 7 not
    assert number_grounding("không có số", ["x"], "") is None


def test_ngram_fidelity_is_the_share_of_answer_8grams_found_in_the_sources():
    src = "một hai ba bốn năm sáu bảy tám chín mười"
    assert ngram_fidelity("một hai ba bốn năm sáu bảy tám chín", [src]) == 1.0
    assert ngram_fidelity("a b c d e f g h i", [src]) == 0.0
    assert ngram_fidelity("ngắn", [src]) is None


def test_pick_by_quantile_orders_from_easy_to_hard():
    rows = [{"id": i, "rouge": i / 10} for i in range(10)]
    picked = pick_by_quantile(rows, 3)
    assert [r["rouge"] for r in picked] == sorted((r["rouge"] for r in picked), reverse=True) and len(picked) == 3
    assert picked[0]["rouge"] >= 0.8 and picked[-1]["rouge"] <= 0.2


def test_latex_escape():
    assert latex_escape("5% & 100_000 #1 {x}") == r"5\% \& 100\_000 \#1 \{x\}"


def test_sample_rows_is_seeded_and_stratified_by_score():
    from evaluate.faithfulness import sample_rows

    rows = [{"id": i, "rouge": i} for i in range(100)]
    a, b = sample_rows(rows, 10, seed=7), sample_rows(rows, 10, seed=7)
    assert a == b and len(a) == 10
    assert sorted({r["rouge"] // 10 for r in a}) == list(range(10))          # one question from each score decile
