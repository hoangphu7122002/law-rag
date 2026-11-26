"""A/B client for speculative decoding: sends the first N dev prompts (retrieved passages + 2 shots, as the QA service builds them) to one vLLM server, greedy, at
concurrency 1 and C; saves the answers and prints decode speed. Run once per server setting, then compare the saved answers.

    PYTHONPATH=src python scripts/serve/spec_bench.py --url http://localhost:8001/v1 --model <path> --out data/runs/spec/base.json [--n 40] [--conc 1,16]
    PYTHONPATH=src python scripts/serve/spec_bench.py --compare data/runs/spec/base.json data/runs/spec/ngram5.json
"""
import argparse
import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def prompts(n: int) -> list[list[dict]]:
    from config import load
    from corpus import QASet
    from generate.prompt import Prompt
    from generate.shots import ShotStore
    from retrieve.retrieval import Retrieval

    cfg = load("configs/exp/A5_v3b.yaml")
    final = Retrieval(cfg)._read()
    prompt, store = Prompt.load(cfg.generate.prompt), ShotStore()
    out = []
    for q in QASet().load("dev"):
        if q.qid in final:
            out.append(prompt.messages(q.text, final[q.qid], store.pick(q.text, 2)))
        if len(out) == n:
            break
    return out


def ask(url: str, model: str, messages: list[dict]) -> tuple[str, int, float]:
    import requests
    t = time.time()
    r = requests.post(f"{url}/chat/completions", json={"model": model, "messages": messages, "temperature": 0, "max_tokens": 1024,
                                                        "chat_template_kwargs": {"enable_thinking": False}}, timeout=600).json()
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], time.time() - t


def run(url, model, msgs, conc):
    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda m: ask(url, model, m), msgs))
    wall = time.time() - t0
    lat = sorted(r[2] for r in res)
    return {"conc": conc, "wall_s": round(wall, 1), "tokens": sum(r[1] for r in res), "tok_per_s": round(sum(r[1] for r in res) / wall, 1),
            "req_per_s": round(len(res) / wall, 2), "p50": round(st.median(lat), 2), "p95": round(lat[int(0.95 * (len(lat) - 1))], 2)}, [r[0] for r in res]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conc", default="1,16")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(f)) for f in a.compare)
        same = sum(p == q for p, q in zip(x["answers"], y["answers"]))
        print(f"identical answers: {same}/{len(x['answers'])}")
        for rx, ry in zip(x["runs"], y["runs"]):
            print(f"conc {rx['conc']}: tok/s {rx['tok_per_s']} -> {ry['tok_per_s']} ({ry['tok_per_s'] / rx['tok_per_s']:.2f}x); p95 {rx['p95']} -> {ry['p95']}")
    else:
        msgs = prompts(a.n)
        run(a.url, a.model, msgs[:4], 1)   # warm-up
        runs, answers = [], None
        for c in [int(x) for x in a.conc.split(",")]:
            r, ans = run(a.url, a.model, msgs, c)
            runs.append(r)
            answers = answers or ans   # answers of the first (lowest-concurrency) pass
            print(json.dumps(r))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"runs": runs, "answers": answers}, open(a.out, "w"), ensure_ascii=False)
