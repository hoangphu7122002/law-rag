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


# ---- extent pairs (2026-11-26): balanced length-direction negatives + hard pairs, mixed with a subsample of the standard pairs
REF = "Căn cứ theo khoản 2 Điều 17 Thông tư X như sau:\n" + "\n".join(f"Dòng {i} " + "nội dung chi tiết " * 12 for i in range(8))


def test_truncate_negative_cuts_at_a_line_boundary_to_about_half_and_skips_one_line_answers():
    import random

    from train.preference.pref_data import truncate_negative
    cut = truncate_negative(REF, random.Random(0))
    assert cut and REF.startswith(cut) and 0.3 * len(REF) <= len(cut) <= 0.65 * len(REF) and not cut.endswith(" ")
    assert truncate_negative("một dòng duy nhất rất dài " * 20, random.Random(0)) is None


def test_extend_negative_appends_a_paragraph_of_the_prompt_context_that_the_answer_does_not_contain():
    import random

    from train.preference.pref_data import extend_negative
    prompt = "ngữ cảnh\n" + "Một đoạn ngữ cảnh khác không có trong câu trả lời " * 6 + "\n" + REF.split("\n")[1]
    ext = extend_negative(REF, prompt, random.Random(0))
    assert ext.startswith(REF.rstrip()) and len(ext) > 1.1 * len(REF) and "đoạn ngữ cảnh khác" in ext
    assert extend_negative(REF, "chỉ có " + REF, random.Random(0)) is None   # nothing new to append


def test_extent_pairs_from_samples_find_good_vs_short_bad_and_good_vs_long_bad():
    from train.preference.pref_data import extent_pairs
    good = REF
    short = "\n".join(REF.split("\n")[:2])
    long = REF + "\n" + "thêm rất nhiều ý không liên quan " * 250
    row = {"qid": 7, "prompt": "p", "reference": REF, "cands": [good, short, long]}
    kinds = {p["source"] for p in extent_pairs(row)}
    assert kinds == {"real_short", "real_long"}
    assert all(p["chosen"] == good for p in extent_pairs(row))


def test_mix_pairs_subsamples_the_standard_pairs_keeps_all_extra_pairs_and_shuffles_deterministically():
    from train.preference.pref_data import mix_pairs
    std = [{"qid": i, "chosen": "c", "rejected": "r", "source": "standard"} for i in range(100)]
    extra = [{"qid": 1000 + i, "chosen": "c", "rejected": "r", "source": "synthetic"} for i in range(30)]
    a = mix_pairs(std, extra, keep_standard=60, seed=1)
    b = mix_pairs(std, extra, keep_standard=60, seed=1)
    assert a == b and len(a) == 90 and sum(p["source"] == "standard" for p in a) == 60
    assert [p["qid"] for p in a] != sorted(p["qid"] for p in a)


def test_best_of_n_row_picks_the_highest_rouge_sample_inside_the_length_band_and_none_when_all_are_bad():
    from train.preference.pref_data import best_of_n_row
    good = REF
    short = "\n".join(REF.split("\n")[:2])
    row = {"qid": 3, "prompt": "p", "reference": REF, "cands": [short, good + " thêm ít", good]}
    out = best_of_n_row(row, min_rouge=0.55)
    assert out == {"qid": 3, "prompt": "p", "completion": good}
    assert best_of_n_row({**row, "cands": [short]}, min_rouge=0.55) is None
    assert best_of_n_row({**row, "reference": "Tôi không thể trả lời câu hỏi này."}, min_rouge=0.55) is None


def test_contrast_pair_sets_a_candidate_citing_the_gold_dieu_against_one_citing_other_dieu():
    from train.preference.pref_data import contrast_pair
    ref = "Căn cứ theo khoản 2 Điều 17 Thông tư X như sau:\nNội dung điều mười bảy rất cụ thể về quy định này."
    right = "Căn cứ theo khoản 2 Điều 17 Thông tư X như sau:\nNội dung điều mười bảy rất cụ thể."
    wrong = "Căn cứ theo khoản 1 Điều 18 Thông tư X như sau:\nNội dung điều mười tám cũng khá cụ thể về quy định này."
    row = {"qid": 5, "prompt": "p", "reference": ref, "cands": [wrong, right]}
    p = contrast_pair(row, min_margin=0.05)
    assert p["chosen"] == right and p["rejected"] == wrong and p["source"] == "cite_contrast" and p["margin"] >= 0.05
    assert contrast_pair({**row, "cands": [right]}, 0.05) is None                       # nothing to contrast with
    assert contrast_pair({**row, "reference": "Tôi không thể trả lời."}, 0.05) is None   # refusal reference
    assert contrast_pair({**row, "reference": "Nội dung không nêu điều nào."}, 0.05) is None   # reference cites no Điều
