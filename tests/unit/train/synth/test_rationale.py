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
