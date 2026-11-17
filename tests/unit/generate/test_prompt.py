from generate import Generator, Prompt
from records import Passage

P = [Passage("a", "50_2024_TT-NHNN", "Điều 17. Nội dung"), Passage("b", "x", "khác")]


def test_baseline_prompt_matches_the_course_notebook():
    text = Prompt.load("baseline").completion("Hỏi gì?", P)
    assert text.startswith("Bạn là một trợ lý thông minh.")
    assert "Ngữ cảnh:\nĐiều 17. Nội dung\nkhác\n\nCâu hỏi: Hỏi gì?" in text
    assert text.endswith("Trả lời ngắn gọn và chính xác nhất có thể:\n")


def test_cancu_prompt_is_chat_with_numbered_context():
    msgs = Prompt.load("cancu").messages("Hỏi gì?", P)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "[1] (50_2024_TT-NHNN)\nĐiều 17. Nội dung" in msgs[1]["content"]


def test_clean_drops_think_blocks():
    assert Generator.clean("<think>x\ny</think>\n Căn cứ ") == "Căn cứ"
