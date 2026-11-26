import sys
from pathlib import Path

from low_score_analysis import failure_stage, summarize_stages


def test_no_article_in_the_reference():
    assert failure_stage(set(), {"A#1"}, "bất kỳ") == "reference cites no article"


def test_gold_article_missing_from_the_passages():
    assert failure_stage({"A#5"}, {"A#1", "A#2"}, "Căn cứ Điều 5") == "gold article not in the passages"


def test_gold_article_retrieved_but_the_answer_cites_another():
    assert failure_stage({"A#5"}, {"A#5", "A#1"}, "Căn cứ Điều 1 quy định") == "retrieved, not cited"


def test_gold_article_retrieved_and_cited():
    assert failure_stage({"A#5"}, {"A#5", "A#1"}, "Căn cứ khoản 2 Điều 5 Thông tư") == "retrieved and cited"


def test_summary_counts_means_and_share_of_low_scores():
    rows = [("retrieved and cited", 0.8), ("retrieved and cited", 0.3), ("retrieved, not cited", 0.2), ("gold article not in the passages", 0.35)]
    s = summarize_stages(rows, low=0.4)
    assert s["retrieved and cited"] == {"n": 2, "mean": 0.55, "low": 1, "low_share": round(1 / 3, 3)}
    assert s["retrieved, not cited"]["low_share"] == round(1 / 3, 3)
