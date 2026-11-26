from train.preference.pref_data import build_pairs

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


def test_extent_bonus_rewards_matching_khoan_and_a_length_close_to_the_reference():
    from train.preference.pref_data import extent_bonus
    ref = "Căn cứ theo khoản 2 Điều 17 Thông tư X như sau: " + "nội dung " * 50
    ok = "Theo khoản 2 Điều 17 Thông tư X: " + "nội dung " * 50
    too_long = "Theo Điều 17 Thông tư X: " + "nội dung " * 200
    no_khoan = "Theo Điều 17 Thông tư X: " + "nội dung " * 50
    assert extent_bonus(ok, ref, 0.15, 0.15) == 0.3
    assert extent_bonus(too_long, ref, 0.15, 0.15) == 0.0
    assert extent_bonus(no_khoan, ref, 0.15, 0.15) == 0.15      # length fine, khoản missing
    assert extent_bonus(no_khoan, "Căn cứ Điều 5 Luật Y: " + "nội dung " * 50, 0.15, 0.15) == 0.3   # the reference names no khoản: nothing to match


def test_build_pairs_with_extent_weights_prefers_the_candidate_with_the_right_extent():
    from train.preference.pref_data import build_pairs
    ref = "Căn cứ theo khoản 2 Điều 17 Thông tư X như sau: " + "nội dung " * 50
    short_wrong = "Theo Điều 17 Thông tư X: nội dung"
    good = "Theo khoản 2 Điều 17 Thông tư X như sau: " + "nội dung " * 50
    rows = [{"qid": 1, "prompt": "p", "reference": ref, "cands": [short_wrong, good]}]
    p = build_pairs(rows, min_margin=0.05, khoan_weight=0.15, length_weight=0.15)
    assert p and p[0]["chosen"] == good and p[0]["rejected"] == short_wrong
