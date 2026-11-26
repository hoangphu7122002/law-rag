from train.preference.grpo import filter_long, score

REF = "Theo Điều 5 quy định về mức phạt tiền đối với hành vi vi phạm hành chính trong lĩnh vực ngân hàng."
PROMPT = "ngữ cảnh khác hoàn toàn"
REFUSE_REF = "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan để trả lời câu hỏi này."


def test_refusal_ref_refusal_pred_full_reward():
    assert score("Các ngữ cảnh được cung cấp không đề cập nội dung này.", REFUSE_REF, PROMPT) == 1.0
    assert score("  DỰA TRÊN CÁC NGỮ CẢNH được cung cấp, không có.  ", REFUSE_REF, PROMPT, cos=0.1) == 1.0


def test_refusal_ref_real_answer_zero():
    assert score(REF, REFUSE_REF, PROMPT) == 0.0


def test_empty_pred_zero():
    assert score("", REFUSE_REF, PROMPT) == 0.0
    assert score("   ", REF, PROMPT) == 0.0


def test_normal_ref_unchanged():
    assert score("Dựa trên các ngữ cảnh được cung cấp, không có.", REF, PROMPT) == 0.0
    assert score(REF, REF, PROMPT) == 1.0
    assert 0.0 < score("Điều 5 quy định mức phạt tiền.", REF, PROMPT) < 1.0
    assert score("Điều 9 khác.", REF, PROMPT) < score("Điều 5 quy định mức phạt tiền.", REF, PROMPT)


def test_filter_long():
    rows = [{"prompt": "a b"}, {"prompt": "a b c d e f"}, {"prompt": "x"}]
    kept, dropped = filter_long(rows, lambda s: len(s.split()), 3)
    assert kept == [rows[0], rows[2]] and dropped == 1
    assert filter_long(rows, lambda s: 3, 3) == (rows, 0)


def test_cite_weight_raises_the_cost_of_a_missing_article_and_defaults_to_the_old_reward():
    wrong = "Điều 9 quy định mức phạt tiền đối với hành vi vi phạm hành chính trong lĩnh vực ngân hàng."
    assert score(wrong, REF, PROMPT) == score(wrong, REF, PROMPT, cite_weight=0.2)
    assert score(wrong, REF, PROMPT, cite_weight=0.4) < score(wrong, REF, PROMPT)
    assert score(REF, REF, PROMPT, cite_weight=0.4) == 1.0           # an exact answer is still a full reward
