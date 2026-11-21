"""Exploratory: can a large LLM (via Mistral Vibe) re-rank the reranker's top candidates better? Dev only.

    box:   PYTHONPATH=src python scripts/llm_rerank.py export --out data/runs/llm_rerank_tasks.json     (dev questions + 10 candidates + gold, from the retrieval cache)
    local: python scripts/llm_rerank.py prompts tasks.json --dir prompts/ --batch 6                      (prompt files WITHOUT gold)
           ... run ~/.claude/skills/vibe/vibe_call.py on each prompt, save the replies ...
           python scripts/llm_rerank.py score tasks.json replies/*.txt                                    (Dieu@1 / @5 / MRR: original order vs LLM order)
The helper is far larger than the 4B answering-model limit: this measures an upper bound on retrieval-side re-ranking, not a deployable system.
"""
import argparse
import json
import re
from pathlib import Path

INSTRUCTION = ("Mỗi câu hỏi dưới đây có một danh sách các điều luật ứng viên. Đáp án chuẩn của câu hỏi sẽ trích dẫn đúng một điều luật trong danh sách. "
               "Hãy xếp các ứng viên theo mức độ chắc chắn là điều luật mà đáp án sẽ trích dẫn (chắc nhất trước). "
               "Trả lời DUY NHẤT một khối JSON dạng {\"<số câu hỏi>\": [<số ứng viên theo thứ tự>, ...], ...}, không giải thích.")


def build_prompt(tasks: dict, text_chars: int = 450) -> str:
    parts = [INSTRUCTION, ""]
    for qid, t in tasks.items():
        parts.append(f"Q{qid}: {t['question']}")
        for c in t["cands"]:
            parts.append(f"  [{c['i']}] ({c['key']}) {c['text'][:text_chars].replace(chr(10), ' ')}")
        parts.append("")
    return "\n".join(parts)


def parse_answer(text: str, tasks: dict) -> dict[int, list[int]]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for k, v in raw.items():
        qid = int(re.sub(r"\D", "", str(k)) or -1)
        if qid in tasks and isinstance(v, list):
            valid = {c["i"] for c in tasks[qid]["cands"]}
            seen, order = set(), []
            for i in v:
                if isinstance(i, int) and i in valid and i not in seen:
                    seen.add(i)
                    order.append(i)
            out[qid] = order
    return out


def _metrics(rank_of: dict[int, list[dict]], tasks: dict) -> dict:
    n = a1 = a5 = 0
    mrr = 0.0
    for qid, cands in rank_of.items():
        gold = set(tasks[qid]["gold"])
        ids = [f"{c['key']}#{c['art']}" for c in cands]
        n += 1
        a1 += bool(ids and ids[0] in gold)
        a5 += any(i in gold for i in ids[:5])
        mrr += next((1 / (r + 1) for r, i in enumerate(ids) if i in gold), 0.0)
    return {"n": n, "art@1": a1 / n if n else 0.0, "art@5": a5 / n if n else 0.0, "art_mrr": mrr / n if n else 0.0}


def score_rankings(tasks: dict, ranks: dict[int, list[int]]) -> dict:
    base, llm = {}, {}
    for qid, t in tasks.items():
        cands = t["cands"]
        base[qid] = cands
        order = ranks.get(qid, [])
        by_i = {c["i"]: c for c in cands}
        first = [by_i[i] for i in order]
        llm[qid] = first + [c for c in cands if c["i"] not in set(order)]   # candidates the helper skipped keep their original order after the chosen ones
    return {"baseline": _metrics(base, tasks), "llm": _metrics(llm, tasks)}


def export(out: Path, k: int = 10) -> None:
    from config import load
    from corpus import QASet
    from corpus.gold import GoldSet
    from evaluate import RetrievalEval
    from retrieve.retrieval import CACHE

    cfg = load("configs/exp/A5_v3b.yaml", ["split=dev", "retrieve.k=10"])
    qs = {q.qid: q for q in QASet().load("dev")}
    gold = GoldSet().load()
    tasks = {}
    for line in (CACHE / f"{cfg.retrieval_key()}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        g = gold.get(r["qid"])
        if r["qid"] not in qs or not g or not g.pairs:
            continue
        cands = []
        for i, p in enumerate(r["passages"][:k], 1):
            key, art = RetrievalEval.passage_keys(p)
            cands.append({"i": i, "key": key, "art": art or "?", "text": p["text"][:600]})
        tasks[r["qid"]] = {"question": qs[r["qid"]].text, "gold": sorted(g.pairs), "cands": cands}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(len(tasks), "dev questions ->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="data/runs/llm_rerank_tasks.json")
    p = sub.add_parser("prompts")
    p.add_argument("tasks")
    p.add_argument("--dir", required=True)
    p.add_argument("--batch", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("tasks")
    s.add_argument("replies", nargs="+")
    a = ap.parse_args()
    if a.cmd == "export":
        export(Path(a.out))
        return
    tasks = {int(k): v for k, v in json.loads(Path(a.tasks).read_text(encoding="utf-8")).items()}
    if a.cmd == "prompts":
        Path(a.dir).mkdir(parents=True, exist_ok=True)
        ids = sorted(tasks)
        for b in range(0, len(ids), a.batch):
            chunk = {q: tasks[q] for q in ids[b: b + a.batch]}
            (Path(a.dir) / f"p{b // a.batch:03d}.txt").write_text(build_prompt(chunk), encoding="utf-8")
        print(len(ids), "questions ->", (len(ids) + a.batch - 1) // a.batch, "prompt files")
        return
    ranks = {}
    for f in a.replies:
        ranks.update(parse_answer(Path(f).read_text(encoding="utf-8"), tasks))
    print(json.dumps({"answered": len(ranks), "of": len(tasks), **score_rankings(tasks, ranks)}, indent=1))


if __name__ == "__main__":
    main()
