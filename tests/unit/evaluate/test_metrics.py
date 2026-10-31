import pytest

from evaluate.metrics import (
    all_lexical,
    bleu_score_simple,
    cosine_similarity_score,
    jaccard_similarity,
    normalize_text,
    rouge_l_score,
    token_overlap_score,
)


def test_normalize_strips_punctuation_and_case():
    assert normalize_text("Căn cứ  Điều 17,\nKhoản 2.") == "căn cứ điều 17 khoản 2"


def test_identical_answers_score_one():
    s = "Căn cứ khoản 2 Điều 17 Thông tư 50/2024/TT-NHNN"
    assert all_lexical(s, s) == pytest.approx({"jaccard": 1, "token_overlap": 1, "bleu1": 1, "rouge_l": 1})


def test_short_prediction_is_penalized_by_brevity():
    true = "a b c d"
    assert bleu_score_simple("a b", true) == pytest.approx(pytest.importorskip("math").exp(1 - 4 / 2))
    assert token_overlap_score("a b", true) == 0.5
    assert jaccard_similarity("a b", true) == 0.5


def test_rouge_l_uses_order():
    assert rouge_l_score("d c b a", "a b c d") == pytest.approx(0.25)


def test_empty_inputs_score_zero():
    assert all_lexical("", "x") == {"jaccard": 0, "token_overlap": 0, "bleu1": 0, "rouge_l": 0}
    assert cosine_similarity_score("", "x", lambda t: [1.0]) == 0.0


def test_cosine_with_fake_embedding():
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert cosine_similarity_score("a", "a", emb.__getitem__) == pytest.approx(1.0)
    assert cosine_similarity_score("a", "b", emb.__getitem__) == pytest.approx(0.0)
