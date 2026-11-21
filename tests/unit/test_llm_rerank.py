import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from llm_rerank import build_prompt, parse_answer, score_rankings

TASKS = {
    1: {"question": "Mức phạt?", "gold": ["A#5"], "cands": [{"i": 1, "key": "A", "art": "1", "text": "Điều 1. x"}, {"i": 2, "key": "A", "art": "5", "text": "Điều 5. y"}, {"i": 3, "key": "B", "art": "2", "text": "Điều 2. z"}]},
    2: {"question": "Khác?", "gold": ["B#2"], "cands": [{"i": 1, "key": "B", "art": "9", "text": "Điều 9. q"}, {"i": 2, "key": "B", "art": "2", "text": "Điều 2. r"}]},
}


def test_prompt_lists_every_question_with_numbered_candidates_and_no_gold():
    p = build_prompt(TASKS)
    assert "Q1" in p and "Q2" in p and "[1]" in p and "Mức phạt?" in p
    assert "A#5" not in p and "gold" not in p.lower()        # the reference article must never be visible to the helper


def test_parse_answer_reads_json_inside_a_reply_and_ignores_bad_indices():
    text = 'Đây là kết quả:\n```json\n{"1": [2, 1, 3], "2": [2, 7, 1]}\n```'
    assert parse_answer(text, TASKS) == {1: [2, 1, 3], 2: [2, 1]}     # 7 is not a candidate of question 2; missing ones are appended by the scorer


def test_scoring_compares_with_the_original_order():
    out = score_rankings(TASKS, {1: [2, 1, 3], 2: [1, 2]})
    assert out["baseline"]["art@1"] == 0.0 and out["llm"]["art@1"] == 0.5     # q1 fixed, q2 not
    assert out["baseline"]["art@5"] == 1.0 and out["llm"]["art@5"] == 1.0
    out2 = score_rankings(TASKS, {})                                           # no answer: keep the original order
    assert out2["llm"] == out2["baseline"]


def test_reorder_puts_the_chosen_candidates_first_and_keeps_the_rest_in_order():
    from llm_rerank import reorder

    ps = ["a", "b", "c", "d", "e", "f"]
    assert reorder(ps, [3, 1], k=4) == ["c", "a", "b", "d"]          # 1-based candidate numbers
    assert reorder(ps, [], k=3) == ["a", "b", "c"]                      # no answer: original order
    assert reorder(ps, [9, 2], k=2) == ["b", "a"]                       # an unknown number is ignored


def test_permute_reverse_and_seeded_shuffle_keep_the_same_passages():
    from llm_rerank import permute

    ps = list("abcdefg")
    assert permute(ps, "reverse", k=5, seed=0) == list("edcba")       # the top-5 reversed, the rest cut
    a = permute(ps, "shuffle", k=5, seed=1)
    assert sorted(a) == sorted(ps[:5]) and a == permute(ps, "shuffle", k=5, seed=1)
    assert permute(ps, "shuffle", k=5, seed=2) != a
