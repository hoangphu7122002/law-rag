"""Teacher-repair runner core (docs/research/teacher-data-plan.md, stage 3): pure and testable; the teacher (prompt -> text) and the tools
(name, args -> text) are injected. Selects wrong trajectories, batches repair requests, verifies and splices the teacher's repairs."""
import json
import time
from pathlib import Path

from train.react.react_data import first_bad_step, grade
from train.react.react_fix import build_batch_prompt, build_repair_request, parse_batch, to_messages, verify_repair

RULES = (
    "QUY TẮC (bắt buộc):\n"
    "- Giữ nguyên các bước trước của học viên; chỉ thay đổi từ bước sai đầu tiên.\n"
    "- Chỉ trả về JSON nghiêm ngặt (một danh sách), không thêm lời giải thích hay văn bản ngoài JSON.\n"
    "- Câu trả lời cuối theo mẫu tham chiếu: \"Căn cứ theo Điều X <Luật> ... như sau:\" rồi trích NGUYÊN VĂN từ điều luật đã truy xuất bằng công cụ.\n"
    "- Tuyệt đối không chép đáp án tham chiếu (sẽ bị kiểm tra trùng 13-gram sau đó); hãy dựa vào điều luật đã truy xuất.\n"
    "- Không bịa số hiệu văn bản hay số điều; chỉ dùng số hiệu/điều có trong câu hỏi, căn cứ đúng hoặc kết quả công cụ.\n\n"
)
OUT_TOKENS_PER_ITEM = 350


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5)


def gold_law_of(pairs) -> str:
    return sorted(pairs)[0].rsplit("#", 1)[0] if pairs else ""


def select_items(rows, qa, gold_pairs, done=frozenset(), limit=None) -> list[dict]:
    """Wrong trajectories (gold Điều never retrieved or answer not ok), one per qid, not already done. qa: qid -> {answer}; gold_pairs: qid -> {'LAW#N'}."""
    items, seen = [], set(done)
    for r in rows:
        q = r["qid"]
        if q in seen or q not in qa:
            continue
        pairs = gold_pairs.get(q) or set()
        gd = {p.rsplit("#", 1)[1] for p in pairs}
        ref = qa[q]["answer"]
        msgs = r["messages"]
        bad = first_bad_step(msgs, gd)
        if bad is None:
            if grade(msgs[-1].get("content") or "", ref, gd)["ok"]:
                continue
            bad = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
            if bad is None:
                continue
        seen.add(q)
        items.append({"qid": q, "messages": msgs, "reference": ref, "gold_dieu": gd, "bad_step": bad,
                      "request": build_repair_request(msgs, ref, gd, gold_law_of(pairs), bad)})
        if limit and len(items) >= limit:
            break
    return items


def done_qids(*paths) -> set[str]:
    out = set()
    for p in paths:
        p = Path(p)
        if p.exists():
            out |= {json.loads(l)["qid"] for l in p.read_text().splitlines() if l.strip()}
    return out


def batch_prompt(batch: list[dict]) -> str:
    return RULES + build_batch_prompt([it["request"] for it in batch])


def plan(items, batch_size=20) -> dict:
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    toks = sum(estimate_tokens(batch_prompt(b)) + OUT_TOKENS_PER_ITEM * len(b) for b in batches)
    return {"items": len(items), "batches": len(batches), "est_tokens": toks}


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_repairs(items, teacher, run_tool, out_path, rej_path, batch_size=20, calls_per_min=20, max_tokens=60_000_000,
                max_calls=None, clock=time.monotonic, sleep=time.sleep, log=print) -> dict:
    stats = {"calls": 0, "tokens": 0, "accepted": 0, "rejected": 0, "stopped": None}
    last = None
    gap = 60.0 / calls_per_min
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt = batch_prompt(batch)
        cost = estimate_tokens(prompt) + OUT_TOKENS_PER_ITEM * len(batch)
        if max_calls is not None and stats["calls"] >= max_calls:
            stats["stopped"] = "max_calls"
            break
        if stats["tokens"] + cost > max_tokens:
            stats["stopped"] = "max_tokens"
            break
        if last is not None and clock() - last < gap:
            sleep(gap - (clock() - last))
        last = clock()
        t0 = time.monotonic()
        try:
            text = teacher(prompt)
        except Exception as e:  # a failed call rejects its items but never kills the run
            log(f"batch {start // batch_size}: teacher error {type(e).__name__}")
            text = ""
        stats["calls"] += 1
        stats["tokens"] += cost
        parsed = parse_batch(text)
        acc = rej = 0
        for i, it in enumerate(batch):
            rep = parsed.get(str(i))
            if rep is None:
                v = {"ok": False, "reason": "no_output"}
            else:
                v = verify_repair(rep, it["reference"], it["gold_dieu"], run_tool, grade)
            if v["ok"]:
                results = [run_tool(a["tool"], {k: x for k, x in a.items() if k != "tool"}) for a in rep["actions"]]
                msgs = to_messages(it["messages"][:it["bad_step"]], rep, results)
                g = grade(rep["answer"], it["reference"], it["gold_dieu"])
                _append(out_path, {"qid": it["qid"], "source": "repair", "messages": msgs, "grade": g})
                acc += 1
            else:
                _append(rej_path, {"qid": it["qid"], "reason": v["reason"], "repair": rep})
                rej += 1
        stats["accepted"] += acc
        stats["rejected"] += rej
        log(f"batch {start // batch_size}: accepted {acc} rejected {rej} elapsed {time.monotonic() - t0:.1f}s")
    return stats
