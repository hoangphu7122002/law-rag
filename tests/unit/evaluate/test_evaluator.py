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
