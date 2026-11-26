"""Retrieval caches whose passages are FULL Điều instead of chunks (no index, no GPU): the quick end-to-end test of 'do not cut long articles'.

Source = the top-15 chunk cache (retrieve.k marker 104). Variants (retrieve.k marker -> rule): 201 first 5 distinct articles of the top-15 chunks, expanded to the
whole article | 202 first 8 | 203 the articles of the top-5 chunks only (<= 5, expanded).  Each passage keeps {pid, doc, text, score} of its best chunk; text = law title
line + the full article (same shape as an indexed chunk).

    PYTHONPATH=src .venv/bin/python scripts/expand_full_article_caches.py          (box; needs data/cpt/articles.jsonl and the marker-104 caches)
"""
import json
import re
from pathlib import Path

from config import load
from evaluate.retrieval_eval import RetrievalEval

n = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())   # noqa: E731


def main():
    A = {}
    for line in open("data/cpt/articles.jsonl", encoding="utf-8"):
        r = json.loads(line)
        k, a = r["key"].split("#", 1)
        A[(n(k), n(a))] = (r["title"], r["text"])

    def path(split, marker):
        cfg = load("configs/exp/A5_v3b.yaml", [f"split={split}", f"retrieve.k={marker}"])
        return Path("data/cache/retrieval") / f"{cfg.retrieval_key()}.jsonl"

    def expand(ps, limit):
        out, seen = [], set()
        for p in ps:
            law, art = RetrievalEval.passage_keys(p)
            key = (n(law), n(art)) if art else ("pid", p.get("pid"))
            if key in seen:
                continue
            seen.add(key)
            if key in A:   # articles.jsonl covers the laws our questions cite; any other chunk stays as it is
                title, text = A[key]
                out.append({**p, "text": f"{title}\n{text}"})
            else:
                out.append(p)
            if len(out) == limit:
                break
        return out

    for split in ("dev", "test"):
        src = path(split, 104)
        if not src.exists():
            print("missing", src)
            continue
        rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()]
        for marker, rule in ((201, lambda ps: expand(ps, 5)), (202, lambda ps: expand(ps, 8)), (203, lambda ps: expand(ps[:5], 5))):
            out = path(split, marker)
            out.write_text("".join(json.dumps({"qid": r["qid"], "passages": rule(r["passages"])}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            tot = sum(len(rule(r["passages"])) for r in rows) / len(rows)
            chars = sum(sum(len(p["text"]) for p in rule(r["passages"])) for r in rows) / len(rows)
            print(f"{split} k={marker}: {len(rows)} q, mean passages {tot:.2f}, mean context chars {chars:.0f} -> {out.name}")


if __name__ == "__main__":
    main()
