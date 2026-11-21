import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from selector_cv import candidate_features, pick_by_prediction


def test_model_candidate_and_quote_candidates_have_distinct_features():
    f_model = candidate_features(kind="model", rank=0, score=0.0, cand="Căn cứ Điều 5 quy định như sau", model_answer="Căn cứ Điều 5 quy định như sau", article="5")
    f_quote = candidate_features(kind="quote", rank=2, score=0.7, cand="Điều 7 nội dung khác hẳn", model_answer="Căn cứ Điều 5 quy định như sau", article="7")
    assert f_model[0] == 1.0 and f_quote[0] == 0.0                    # is_model
    assert f_quote[1] == 2 and f_quote[2] == 0.7                       # rank, reranker score
    assert f_model[3] == 1.0 and f_quote[3] < 1.0                      # rouge against the model answer
    assert f_model[4] == 1.0 and f_quote[4] == 0.0                     # the candidate's Điều is the one the model cited


def test_pick_by_prediction_takes_the_highest_predicted_candidate_per_question():
    preds = {1: [0.2, 0.9, 0.1], 2: [0.5, 0.4]}
    truth = {1: [0.3, 0.8, 0.1], 2: [0.6, 0.9]}
    assert pick_by_prediction(preds, truth) == (0.8 + 0.6) / 2
