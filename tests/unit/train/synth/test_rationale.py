from train.synth.rationale import assemble, rationale_request, to_think_prompt, valid_rationale

PROMPT = ("<|im_start|>system\nBạn là trợ lí.<|im_end|>\n<|im_start|>user\nNgữ cảnh:\n- Điều 17. Nội dung.\n### Câu hỏi :\nPhạt bao nhiêu?\n\n### Trả lời :<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
REF = "Căn cứ khoản 1 Điều 17 Thông tư X như sau:\nPhạt 100.000 đồng."


def test_to_think_prompt_opens_a_thinking_turn_instead_of_the_empty_think_block():
    out = to_think_prompt(PROMPT)
    assert out.endswith("<|im_start|>assistant\n<think>\n") and "</think>" not in out


def test_assemble_puts_the_rationale_in_a_think_block_before_the_gold_answer():
    row = assemble({"qid": 1, "prompt": PROMPT, "completion": REF}, "Câu hỏi hỏi mức phạt nên chọn khoản 1 Điều 17.")
    assert row["prompt"].endswith("<think>\n")
    assert row["completion"] == "Câu hỏi hỏi mức phạt nên chọn khoản 1 Điều 17.\n</think>\n\n" + REF and row["qid"] == 1


def test_valid_rationale_rejects_answer_leaks_wrong_articles_and_bad_lengths():
    ok = "Câu hỏi hỏi mức phạt. Khoản 1 Điều 17 nêu đúng mức phạt, nên trích khoản này."
    assert valid_rationale(ok, REF)
    assert not valid_rationale("Ngắn.", REF)
    assert not valid_rationale(ok + " Theo đáp án chuẩn thì đúng.", REF)           # leaks that the answer was known
    assert not valid_rationale("Câu hỏi hỏi mức phạt nên theo Điều 99 là đúng, trích khoản này để trả lời đầy đủ.", REF)   # a Điều the reference does not cite
    assert not valid_rationale(ok * 30, REF)                                        # too long


def test_rationale_request_shows_context_question_and_reference_and_forbids_mentioning_it():
    msg = rationale_request(PROMPT, REF)
    assert "Phạt bao nhiêu?" in msg and "Phạt 100.000 đồng." in msg and "không nhắc" in msg.lower()


def test_split_think_separates_a_complete_thought_from_the_answer_and_rejects_truncated_ones():
    from train.synth.rationale import split_think
    assert split_think("Ta cần tìm khoản đúng.\nKhoản 1 phù hợp.\n</think>\n\nCăn cứ khoản 1 Điều 17") == ("Ta cần tìm khoản đúng.\nKhoản 1 phù hợp.", "Căn cứ khoản 1 Điều 17")
    assert split_think("suy nghĩ chưa xong, bị cắt") is None
    assert split_think("</think>\n\nchỉ có đáp án") is None


def test_accept_trace_needs_the_gold_article_in_the_final_answer_and_enough_overlap():
    from train.synth.rationale import accept_trace
    good = "Căn cứ khoản 1 Điều 17 Thông tư X như sau:\nPhạt 100.000 đồng."
    assert accept_trace("t" * 200, good, REF, min_rouge=0.5)
    assert not accept_trace("t" * 200, "Căn cứ Điều 18 như sau:\nPhạt 100.000 đồng.", REF, min_rouge=0.5)   # another Điều
    assert not accept_trace("t" * 200, "Căn cứ khoản 1 Điều 17 như sau:\n" + "nội dung khác hoàn toàn không liên quan " * 12, REF, min_rouge=0.5)
    assert not accept_trace("ngắn", good, REF, min_rouge=0.5)                                    # trivial thought
    assert not accept_trace("t" * 20000, good, REF, min_rouge=0.5)                                # runaway thought


def test_accept_trace_for_a_refusal_reference_wants_a_refusal_answer():
    from train.synth.rationale import accept_trace
    refusal = "Tôi không thể trả lời câu hỏi này vì ngữ cảnh không có thông tin."
    assert accept_trace("t" * 200, refusal, refusal)
    assert not accept_trace("t" * 200, "Căn cứ khoản 1 Điều 17 như sau: Phạt.", refusal)


def test_valid_thinking_accepts_a_longer_stepwise_thought_but_not_leaks_or_foreign_articles():
    from train.synth.rationale import valid_thinking
    good = ("Câu hỏi hỏi mức phạt khi không có bảo hiểm. Xem ngữ cảnh: Điều 17 có ba khoản. Khoản 1 nói về người không có bảo hiểm bắt buộc, đúng với câu hỏi. "
            "Khoản 2 nói về giấy đăng ký xe nên loại. Khoản 3 là tái phạm, câu hỏi không nhắc. Vậy chọn khoản 1 Điều 17 và trích nguyên văn khoản này, không cần thêm khoản khác. " * 2)
    assert valid_thinking(good, REF)
    assert not valid_thinking(good + " Theo đáp án chuẩn thì đúng.", REF)
    assert valid_thinking(good + " Điều 99 nói về việc khác nên loại.", REF)                  # eliminating another Điều is fine
    assert not valid_thinking(good.replace("Điều 17", "điều này"), REF)                       # but the reference's Điều must be named
    assert not valid_thinking("Quá ngắn.", REF)


def test_is_vietnamese_rejects_english_thoughts():
    from train.synth.rationale import is_vietnamese
    assert is_vietnamese("Câu hỏi hỏi về mức phạt, và Điều 17 là điều khoản của luật này nên ta chọn khoản 1.")
    assert not is_vietnamese("Thinking Process: the question asks about the fine, so we should pick clause 1 of Article 17 and quote it.")


def test_pick_thought_takes_the_shortest_thought_among_near_best_answers_and_skips_bad_ones():
    from train.synth.rationale import pick_thought
    vi = "Câu hỏi hỏi mức phạt, và Điều 17 là điều khoản của luật này nên chọn khoản 1. "
    good = "Căn cứ khoản 1 Điều 17 Thông tư X như sau:\nPhạt 100.000 đồng."
    cands = [(vi * 6, good), (vi * 3, good), ("Thinking Process: " + "english " * 40, good), (vi * 2, "Căn cứ Điều 18 như sau:\nPhạt 100.000 đồng.")]
    assert pick_thought(cands, REF, min_rouge=0.5) == vi * 3          # shortest valid; English and wrong-Điều candidates ignored
    assert pick_thought([(vi * 3, "Căn cứ Điều 18: khác.")], REF) is None
    assert pick_thought([("t" * 3000, good)], REF) is None            # too long for a deployable thought (max 1800 chars)
