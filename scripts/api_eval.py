"""Send dev questions to the running QA API and score the answers (parity check with the offline run + end-to-end latency).

    PYTHONPATH=src python scripts/api_eval.py [--url http://localhost:8080] [--n 30] [--workers 4]
Dev only. Latency = wall time of one POST /ask (retrieve + rerank + generate), measured by the client.
"""
import argparse
import json
import statistics as st
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def summarize(rows: list[dict]) -> dict:
    ok = sorted((r for r in rows if "error" not in r), key=lambda r: r["seconds"])
    s = [r["seconds"] for r in ok]
    return {"n": len(ok), "errors": len(rows) - len(ok), "rouge_l": round(st.mean(r["rouge"] for r in ok), 4) if ok else None,
            "mean_s": round(st.mean(s), 2) if s else None, "p50_s": s[len(s) // 2] if s else None,
            "p95_s": s[min(len(s) - 1, int(len(s) * 0.95))] if s else None}


def ask(url: str, question: str) -> dict:
    req = urllib.request.Request(f"{url}/ask", json.dumps({"question": question}).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out | {"seconds": time.perf_counter() - t}


def main() -> None:
    from corpus import QASet
    from evaluate.metrics import rouge_l_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    qs = QASet().load("dev")[: a.n]

    def one(q):
        try:
            out = ask(a.url, q.text)
            return {"qid": q.qid, "rouge": rouge_l_score(out["answer"], q.answer), "seconds": out["seconds"], "server": out["timings"]}
        except Exception as e:  # noqa: BLE001
            return {"qid": q.qid, "error": repr(e)[:200], "seconds": 0.0}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, qs))
    print(json.dumps(summarize(rows)))
    srv = [r["server"] for r in rows if "server" in r]
    if srv:
        print("server-side mean s:", {k: round(st.mean(x[k] for x in srv), 3) for k in srv[0]})


if __name__ == "__main__":
    main()
