"""Rationales for thinking-mode SFT (STaR-style rationalisation): a stronger model, shown the context, the question AND the reference answer, writes the short reasoning that
leads to that answer as if the answer were unknown; the training row is  prompt + "<think>\n" -> rationale + "\n</think>\n\n" + reference answer.

    to_think_prompt(prompt)            non-thinking prompt (ends in an empty think block) -> prompt that opens a thinking turn
    rationale_request(prompt, ref)     the user message for the rationaliser
    valid_rationale(text, ref)         length, no leak of the answer key, and no Điều the reference does not cite
    assemble(row, rationale)           the thinking SFT row
"""
import re

from evaluate.faithfulness import dieu_numbers

EMPTY_THINK = re.compile(r"<think>\s*</think>\s*$")
LEAK = re.compile(r"đáp án|câu trả lời (chuẩn|mẫu|đã cho|cho sẵn)|theo (đáp án|câu trả lời chuẩn)|tham chiếu", re.I)
MIN_CHARS, MAX_CHARS = 60, 1200


def to_think_prompt(prompt: str) -> str:
    return EMPTY_THINK.sub("<think>\n", prompt.rstrip("\n")) if EMPTY_THINK.search(prompt.rstrip("\n")) else prompt


def rationale_request(prompt: str, reference: str) -> str:
    m = re.search(r"### Câu hỏi :\s*(.*?)\s*### Trả lời", prompt, re.S)
    question = m.group(1) if m else ""
    ctx = prompt.split("<|im_start|>user\n", 1)[-1].split("### Câu hỏi", 1)[0]
    return ("Dưới đây là ngữ cảnh pháp lý, một câu hỏi và câu trả lời đúng. Hãy viết phần SUY LUẬN ngắn gọn (3-6 câu, tiếng Việt) như thể bạn chưa biết câu trả lời: "
            "(1) xác định văn bản, Điều, khoản trong ngữ cảnh trả lời câu hỏi; (2) nêu ngắn gọn vì sao không chọn các Điều hoặc khoản gần nghĩa khác; "
            "(3) nói cần trích nguyên văn toàn bộ khoản hay chỉ một số điểm. Nếu ngữ cảnh không có thông tin thì nói rõ. "
            "Không nhắc tới việc đã biết trước câu trả lời, không viết lại câu trả lời, chỉ viết suy luận.\n\n"
            f"### Ngữ cảnh\n{ctx.strip()}\n\n### Câu hỏi\n{question}\n\n### Câu trả lời đúng (chỉ để định hướng suy luận, không nhắc tới)\n{reference}")


def valid_rationale(text: str, reference: str) -> bool:
    t = text.strip()
    if not (MIN_CHARS <= len(t) <= MAX_CHARS) or LEAK.search(t):
        return False
    return dieu_numbers(t) <= (dieu_numbers(reference) or dieu_numbers(t))


def assemble(row: dict, rationale: str) -> dict:
    return {"qid": row["qid"], "prompt": to_think_prompt(row["prompt"]), "completion": rationale.strip() + "\n</think>\n\n" + row["completion"]}
