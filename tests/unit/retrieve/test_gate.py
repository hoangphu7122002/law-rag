from retrieve.gate import expand_context, features, select_threshold


def P(doc, art, score):
    return {"doc": doc, "text": f"Luật X\nĐiều {art}. Tiêu đề\nnội dung", "score": score}


def test_features_have_a_fixed_length_and_see_the_score_shape():
    ps = [P("a_b", i, 1.0 - i * 0.1) for i in range(1, 11)]
    f = features("Luật nào quy định?", ps)
    assert len(f) == len(features("câu hỏi khác", ps[:3]))          # short lists are padded
    assert f[0] == 0.9 and f[5] > 0                                 # top-1 score and the gap top1-top2


def test_a_law_named_in_the_question_but_absent_from_the_top_docs_is_visible_to_the_gate():
    ps = [P("50_2024_TT-NHNN", 1, 0.9)] * 5
    present = features("theo Thông tư 50/2024/TT-NHNN thì sao", ps)
    absent = features("theo Thông tư 99/2020/TT-NHNN thì sao", ps)
    assert present != absent


def test_expand_context_takes_more_passages_only_for_flagged_questions():
    ps = list(range(20))
    assert expand_context(ps, flagged=False, base_k=5, wide_k=15) == ps[:5]
    assert expand_context(ps, flagged=True, base_k=5, wide_k=15) == ps[:15]


def test_threshold_flags_the_requested_share_of_questions():
    probs = [i / 100 for i in range(100)]
    t = select_threshold(probs, share=0.2)
    assert sum(p >= t for p in probs) == 20
