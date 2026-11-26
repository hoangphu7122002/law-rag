"""Rationales for thinking-mode SFT (STaR-style rationalisation): a stronger model, shown the context, the question AND the reference answer, writes the short reasoning that
leads to that answer as if the answer were unknown; the training row is  prompt + "<think>\n" -> rationale + "\n</think>\n\n" + reference answer.

    to_think_prompt(prompt)            non-thinking prompt (ends in an empty think block) -> prompt that opens a thinking turn
    rationale_request(prompt, ref)     the user message for the rationaliser
    valid_rationale(text, ref)         length, no leak of the answer key, and names every Điều the reference cites
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
    return dieu_numbers(reference) <= dieu_numbers(t)   # names every cited Điều; others may appear (the request asks to rule out near misses)


def assemble(row: dict, rationale: str) -> dict:
    return {"qid": row["qid"], "prompt": to_think_prompt(row["prompt"]), "completion": rationale.strip() + "\n</think>\n\n" + row["completion"]}


def split_think(text: str) -> tuple[str, str] | None:
    """Model output of a thinking turn (the opening <think> is in the prompt): (thought, final answer), None when the thought was cut off or empty."""
    if "</think>" not in text:
        return None
    thought, answer = text.split("</think>", 1)
    thought, answer = thought.strip(), answer.strip()
    return (thought, answer) if thought and answer else None


def accept_trace(thought: str, final: str, reference: str, min_rouge: float = 0.5, min_thought: int = 120, max_thought: int = 1800) -> bool:
    """A self-generated thought is kept when it ends in an answer that cites every Điều of the reference (and none the reference lacks) and overlaps it (ROUGE-L)."""
    from evaluate.metrics import rouge_l_score
    if not (min_thought <= len(thought) <= max_thought):
        return False
    from train.preference.grpo import is_refusal
    if is_refusal(reference):      # no-answer rows: the reasoning must end in a refusal too
        return is_refusal(final)
    gold, cited = dieu_numbers(reference), dieu_numbers(final)
    if gold and cited != gold:
        return False
    return rouge_l_score(final, reference) >= min_rouge


def valid_thinking(text: str, reference: str, min_chars: int = 400, max_chars: int = 3500) -> bool:
    """Longer step-by-step thought (teacher-written): no answer-key leak and it names every Điều of the reference (other Điều may be mentioned: that is the elimination)."""
    t = text.strip()
    if not (min_chars <= len(t) <= max_chars) or LEAK.search(t):
        return False
    return dieu_numbers(reference) <= dieu_numbers(t)


VI_WORDS = ("của", "và", "là", "các", "không", "được", "điều", "khoản", "này", "câu hỏi")


def is_vietnamese(text: str) -> bool:
    """Cheap language check: a Vietnamese thought uses several of the most frequent Vietnamese function words."""
    t = text.lower()
    return sum(w in t for w in VI_WORDS) >= 4


def pick_thought(cands: list[tuple[str, str]], reference: str, min_rouge: float = 0.5, tol: float = 0.05) -> str | None:
    """Among (thought, final answer) pairs keep the Vietnamese, accepted ones; the best final-answer ROUGE-L sets the bar (within `tol`) and the SHORTEST thought above it wins
    (short thoughts cost little latency at inference)."""
    from evaluate.metrics import rouge_l_score
    ok = [(t, rouge_l_score(f, reference)) for t, f in cands if is_vietnamese(t) and accept_trace(t, f, reference, min_rouge)]
    if not ok:
        return None
    best = max(r for _, r in ok)
    return min((t for t, r in ok if r >= best - tol), key=len)
