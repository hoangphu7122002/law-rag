"""Trajectory data for the ReAct student, datamate style (docs/notes/16-react-training.md): S1 = the student's own good trajectories,
S2 = the teacher repairs a wrong one FROM ITS FIRST BAD STEP (the prefix is the student's, the continuation the teacher's).

Pure helpers (no GPU / API): grade a final answer, find the first bad step, cut a conversation into one training row per assistant turn
(prompt = everything before the turn, completion = the turn; loss only on completions).
"""
import re

from evaluate.metrics import rouge_l_score

ROUGE_OK = 0.6          # a final answer counts as good when rougeL against the course reference reaches this AND it cites the gold Điều


def cited(text: str) -> set[str]:
    return set(re.findall(r"[Đđ]iều\s+(\d+[a-zđ]?)", text or ""))


def grade(answer: str, reference: str, gold_dieu: set[str]) -> dict:
    r = rouge_l_score(answer, reference)
    cites = bool(cited(answer) & set(gold_dieu)) if gold_dieu else True
    return {"rouge": r, "cites_gold": cites, "ok": r >= ROUGE_OK and cites}


def _seen(text: str, gold_dieu: set[str]) -> bool:
    return any(re.search(rf"[Đđ]iều\s+{re.escape(n)}\b", text or "") for n in gold_dieu)


def first_bad_step(msgs: list[dict], gold_dieu: set[str]) -> int | None:
    """Index of the first assistant message when the gold Điều never reached the model (not in the prompt nor in any tool result),
    else None: retrieval was fine, any remaining error is in the answer itself (graded separately)."""
    if any(_seen(m["content"], gold_dieu) for m in msgs if m["role"] in ("user", "tool")):
        return None
    return next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)


def to_turn_rows(msgs: list[dict]) -> list[dict]:
    return [{"prompt": msgs[:i], "completion": m} for i, m in enumerate(msgs) if m["role"] == "assistant"]
