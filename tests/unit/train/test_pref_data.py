from train.pref_data import build_pairs

REF = "Căn cứ Điều 5 quy định về mức phạt tiền đối với hành vi vi phạm hành chính trong lĩnh vực ngân hàng."
GOOD = "Căn cứ Điều 5 quy định về mức phạt tiền đối với hành vi vi phạm hành chính trong lĩnh vực ngân hàng."
MID = "Căn cứ Điều 5 quy định mức phạt tiền đối với hành vi vi phạm."
BAD = "Hôm nay trời đẹp và tôi đi chơi công viên với bạn bè."


def row(cands, ref=REF):
    return {"qid": 1, "prompt": "ctx", "reference": ref, "cands": cands}


def test_pair_is_best_vs_worst_by_reward():
    p = build_pairs([row([MID, BAD, GOOD])], min_margin=0.1)
    assert len(p) == 1 and p[0]["chosen"] == GOOD and p[0]["rejected"] == BAD and p[0]["prompt"] == "ctx"


def test_close_candidates_are_skipped():
    assert build_pairs([row([GOOD, GOOD + " "])], min_margin=0.1) == []


def test_refusal_reference_rows_are_skipped():
    refusal = "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan."
    assert build_pairs([row([GOOD, BAD], ref=refusal)], min_margin=0.1) == []


def test_empty_candidates_never_chosen_or_rejected_as_duplicates():
    p = build_pairs([row(["", GOOD, GOOD])], min_margin=0.1)
    assert all(x["chosen"] and x["rejected"] != x["chosen"] for x in p)


def test_cite_weight_changes_which_candidate_wins():
    other_article = "Căn cứ Điều 9 quy định về mức phạt tiền đối với hành vi vi phạm hành chính trong lĩnh vực ngân hàng."
    right_short = "Căn cứ Điều 5 quy định mức phạt tiền."
    p = build_pairs([row([other_article, right_short, BAD])], min_margin=0.05, cite_weight=0.9)
    assert p and p[0]["chosen"] == right_short
