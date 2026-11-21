"""Closed-loop load test of the QA API: U virtual users, each sends questions back to back for --seconds.

    PYTHONPATH=src python scripts/load_test.py --url http://localhost:8080 --users 1,4,8,16,32 --seconds 60 [--out data/runs/load.jsonl]
Per user level one JSON line: ok, errors, 429s, requests/s, TTFT and total latency p50/p95 (client side, via /ask/stream).
Questions are dev questions cycled (answers are not scored here; see scripts/api_eval.py for quality).
"""
import argparse
import asyncio
import json
import statistics as st
import time


def parse_sse_line(line: str):
    if not line.strip():
        return None
    key, _, val = line.partition(": ")
    return (key, json.loads(val)) if key == "data" else (key, val)


def pct(xs: list[float], q: float):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * q))], 3) if xs else None


def summarize_load(recs: list[dict], wall: float, users: int) -> dict:
    ok = [r for r in recs if "total" in r]
    ttft, total = [r["ttft"] for r in ok], [r["total"] for r in ok]
    return {"users": users, "ok": len(ok), "errors": sum(1 for r in recs if "status" in r and r["status"] != 429),
            "rejected_429": sum(1 for r in recs if r.get("status") == 429),
            "rps": round(len(ok) / wall, 3) if wall > 0 else 0.0,
            "ttft_p50": pct(ttft, 0.5), "ttft_p95": pct(ttft, 0.95), "total_p50": pct(total, 0.5), "total_p95": pct(total, 0.95),
            "total_mean": round(st.mean(total), 3) if total else None}


async def one_request(client, url: str, question: str) -> dict:
    t0 = time.perf_counter()
    ttft = None
    try:
        async with client.stream("POST", f"{url}/ask/stream", json={"question": question}, timeout=300) as r:
            if r.status_code != 200:
                return {"status": r.status_code}
            event = None
            async for line in r.aiter_lines():
                p = parse_sse_line(line)
                if p is None:
                    continue
                if p[0] == "event":
                    event = p[1]
                elif event == "token" and ttft is None:
                    ttft = time.perf_counter() - t0
                elif event == "error":
                    return {"status": 503}
        return {"ttft": ttft if ttft is not None else time.perf_counter() - t0, "total": time.perf_counter() - t0}
    except Exception:  # noqa: BLE001
        return {"status": -1}


async def run_level(url: str, questions: list[str], users: int, seconds: float) -> dict:
    import httpx

    recs: list[dict] = []
    stop = time.perf_counter() + seconds
    counter = iter(range(10**9))
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=users + 4)) as client:
        async def user():
            while time.perf_counter() < stop:
                recs.append(await one_request(client, url, questions[next(counter) % len(questions)]))

        t0 = time.perf_counter()
        await asyncio.gather(*(user() for _ in range(users)))
        wall = time.perf_counter() - t0
    return summarize_load(recs, wall, users)


def main() -> None:
    from corpus import QASet

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--users", default="1,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--n", type=int, default=100, help="dev questions to cycle")
    ap.add_argument("--tag", default="", help="label stored with every line (e.g. the GPU layout)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    qs = [q.text for q in QASet().load("dev")[: a.n]]
    for u in (int(x) for x in a.users.split(",")):
        row = {"tag": a.tag} | asyncio.run(run_level(a.url, qs, u, a.seconds))
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
