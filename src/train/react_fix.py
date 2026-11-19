"""Teacher-fix tooling for ReAct training data (docs/notes/16-react-training.md, S2): pure helpers, the teacher (Haiku) and the tools
are called OUTSIDE this module. A wrong student trajectory is repaired from its first bad step: the teacher returns up to 2 tool calls
plus a final answer as strict JSON; we verify it against the tools and the reference, then splice it into the student's prefix."""
import json
import re

MAX_ACTIONS = 2


def _json_candidates(text: str):
    text = text or ""
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            yield dec.raw_decode(text[m.start():])[0]
        except ValueError:
            continue


def _valid_action(a) -> bool:
    if not isinstance(a, dict):
        return False
    if a.get("tool") == "search_law":
        return isinstance(a.get("query"), str) and bool(a["query"].strip())
    if a.get("tool") == "get_article":
        return isinstance(a.get("law"), str) and bool(a["law"].strip()) and isinstance(a.get("dieu"), int) and not isinstance(a["dieu"], bool)
    return False


def _valid_repair(r) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("actions"), list) and len(r["actions"]) <= MAX_ACTIONS
            and all(_valid_action(a) for a in r["actions"]) and isinstance(r.get("answer"), str) and bool(r["answer"].strip()))


def parse_repair(text: str) -> dict | None:
    return next((c for c in _json_candidates(text) if _valid_repair(c)), None)


def parse_batch(text: str) -> dict[str, dict]:
    """{item id: repair} for the valid items of a JSON list answer (invalid items are dropped)."""
    for c in _json_candidates(text):
        if isinstance(c, list):
            out = {}
            for it in c:
                if isinstance(it, dict) and "id" in it:
                    r = {k: it.get(k) for k in ("actions", "answer")}
                    if _valid_repair(r):
                        out[str(it["id"])] = r
            return out
    return {}


NGRAM = 13


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


def copy_share(answer: str, reference: str, n: int = NGRAM) -> float:
    """Share of the answer's word n-grams (lower-cased, punctuation stripped) that also occur in the reference; 0.0 if the answer has < n words."""
    a, r = _words(answer), _words(reference)
    grams = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    if not grams:
        return 0.0
    ref = {tuple(r[i:i + n]) for i in range(len(r) - n + 1)}
    return sum(g in ref for g in grams) / len(grams)


def verify_repair(repair, reference: str, gold_dieu: set[str], run_tool, grade, max_copy: float = 0.5) -> dict:
    if not _valid_repair(repair):
        return {"ok": False, "reason": "invalid"}
    results = [run_tool(a["tool"], {k: v for k, v in a.items() if k != "tool"}) for a in repair["actions"]]
    if not any(re.search(rf"[Đđ]iều\s+{re.escape(n)}\b", t or "") for t in results for n in gold_dieu):
        return {"ok": False, "reason": "gold_not_retrieved"}
    if not grade(repair["answer"], reference, gold_dieu)["ok"]:
        return {"ok": False, "reason": "answer_not_ok"}
    if copy_share(repair["answer"], reference) > max_copy:
        return {"ok": False, "reason": "copies_reference"}
    return {"ok": True, "reason": "ok"}


TOOL_CHARS = 300        # a tool result in the prefix is cut to this many chars in the prompt
SCHEMA = '{"actions":[{"tool":"search_law","query":"..."} | {"tool":"get_article","law":"...","dieu":N}], "answer":"..."}'


def _render(m: dict) -> str:
    if m["role"] == "tool":
        return f"[kết quả công cụ] {(m.get('content') or '')[:TOOL_CHARS]}"
    calls = "; ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m.get("tool_calls") or [])
    return f"[{m['role']}] {(m.get('content') or '')[:TOOL_CHARS]}" + (f" gọi: {calls}" if calls else "")


def build_repair_request(messages: list[dict], reference: str, gold_dieu: set[str], gold_law: str, bad_step: int) -> str:
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    prefix = "\n".join(_render(m) for m in messages[:bad_step] if m["role"] != "system")
    dieu = ", ".join(f"Điều {n}" for n in sorted(gold_dieu))
    return (f"Câu hỏi: {question}\n\nHội thoại của học viên cho tới bước sai:\n{prefix}\n\n"
            f"Căn cứ đúng: {dieu} của văn bản {gold_law}.\nĐáp án tham chiếu của khóa học: {reference}\n\n"
            f"Hãy viết các bước TIẾP THEO đã sửa: tối đa {MAX_ACTIONS} lời gọi công cụ (search_law(query), get_article(law, dieu)) để lấy đúng căn cứ, "
            "rồi câu trả lời cuối theo dạng 'Căn cứ Điều X ... quy định ...'. Không chép nguyên văn đáp án tham chiếu. "
            f"Chỉ trả về JSON nghiêm ngặt: {SCHEMA}")


def build_batch_prompt(requests: list[str]) -> str:
    items = "\n\n".join(f"=== Mục id={i} ===\n{r}" for i, r in enumerate(requests))
    return (f"Dưới đây là {len(requests)} yêu cầu sửa. Trả lời bằng MỘT danh sách JSON, mỗi mục một đối tượng "
            f'{{"id": <id của mục>, "actions": [...], "answer": "..."}} theo đúng định dạng của từng yêu cầu.\n\n{items}')


def to_messages(prefix: list[dict], repair: dict, tool_results: list[str]) -> list[dict]:
    """prefix + the teacher's repaired turns, in react.py's message format (assistant tool_calls, tool messages, final assistant answer)."""
    out = list(prefix)
    base = sum(1 for m in prefix for _ in m.get("tool_calls") or [])
    for i, (a, res) in enumerate(zip(repair["actions"], tool_results)):
        cid = f"fix{base + i}"
        args = {k: v for k, v in a.items() if k != "tool"}
        out.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": a["tool"], "arguments": json.dumps(args, ensure_ascii=False)}}]})
        out.append({"role": "tool", "tool_call_id": cid, "content": res})
    out.append({"role": "assistant", "content": repair["answer"]})
    return out
