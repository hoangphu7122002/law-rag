import pytest

from evaluate import Evaluator, PairedBootstrap


def test_law_hit_matches_file_stem_of_cited_law():
    assert Evaluator.law_hit("50/2024/TT-NHNN", ["x", "50_2024_TT-NHNN"])
    assert not Evaluator.law_hit("50/2024/TT-NHNN", ["51_2024_TT-NHNN"])
    assert not Evaluator.law_hit("", ["anything"])


def test_bootstrap_sees_a_clear_gain_and_no_gain():
    a = {i: 0.2 for i in range(100)}
    better = PairedBootstrap(n=2000).compare(a, {i: 0.3 + 0.01 * (i % 3) for i in range(100)})
    assert better["lo"] > 0 and better["diff"] == pytest.approx(0.11, abs=0.01)
    same = PairedBootstrap(n=2000).compare(a, a)
    assert same["diff"] == 0 and same["p_b_not_better"] == 1.0


def test_law_hit_ignores_web_page_suffix_and_case():
    assert Evaluator.law_hit("2124/QĐ-TTg", ["2124_QĐ-TTg__285123"])


def test_retrieval_eval_ranks_doc_and_article():
    from corpus.gold import Gold
    from evaluate import RetrievalEval

    gold = {1: Gold(1, frozenset({"5/2024/TT-NHNN"}), frozenset({"5/2024/TT-NHNN#3"}), "5/2024/TT-NHNN")}
    ps = [{"doc": "9_2020_ND-CP", "text": "Điều 3. X"}, {"doc": "5_2024_TT-NHNN", "text": "Thông tư 5\nĐiều 2. Y"},
          {"doc": "5_2024_TT-NHNN", "text": "Thông tư 5\nĐiều 3. Z"}]
    row = RetrievalEval(gold).score_one(1, ps)
    assert (row["doc@1"], row["doc@3"], row["doc_mrr"]) == (0, 1, 0.5)
    assert (row["art@1"], row["art@3"]) == (0, 1)
    assert RetrievalEval.summary([row])["art_mrr"] == round(1 / 3, 4)
