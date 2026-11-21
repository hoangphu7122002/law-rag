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


def meets_slo(row: dict, slo_p95: float) -> bool:
    """SLO: p95 total latency under the bound, under 1 % errors, and nothing rejected (429)."""
    n = row["ok"] + row["errors"]
    return bool(row["ok"]) and row["rejected_429"] == 0 and row["errors"] <= 0.01 * n and row["total_p95"] is not None and row["total_p95"] <= slo_p95


def find_max_users(measure, start: int, limit: int, slo_p95: float):
    """Largest user count that meets the SLO: double from `start` until it fails (or hits `limit`), then bisect the gap.
    `measure(users) -> row`. Returns (best, all rows in call order); best = 0 when even `start` fails."""
    rows, good, bad, u = [], 0, None, start
    while True:
        row = measure(u)
        rows.append(row)
        if meets_slo(row, slo_p95):
            good = u
            if u >= limit:
                return good, rows
            u = min(u * 2, limit)
        else:
            bad = u
            break
    lo, hi = good, bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        row = measure(mid)
        rows.append(row)
        if meets_slo(row, slo_p95):
            lo = mid
        else:
            hi = mid
    return lo, rows


def parse_smi(text: str) -> dict[int, float]:
    """`nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits` -> {gpu: util %}"""
    out = {}
    for ln in text.strip().splitlines():
        i, u = (x.strip() for x in ln.split(","))
        out[int(i)] = float(u)
    return out


def mean_by_gpu(samples: list[dict[int, float]]) -> dict[str, float]:
    if not samples:
        return {}
    return {f"gpu{g}_util": round(st.mean(s[g] for s in samples if g in s), 1) for g in sorted(samples[0])}


def level_problem(row: dict):
    """A level that did not really exercise the stack: nothing succeeded, or every GPU stayed idle while requests ran."""
    if row["ok"] == 0:
        return f"no request succeeded ({row['errors']} errors, {row['rejected_429']} rejected)"
    utils = [v for k, v in row.items() if k.endswith("_util")]
    if utils and max(utils) < 5:
        return f"all GPUs idle (max mean util {max(utils)} %) although {row['ok']} requests completed: requests are not reaching the LLM"
    return None


async def sample_gpus(samples: list, stop: asyncio.Event) -> None:
    import subprocess

    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                samples.append(parse_smi(r.stdout))
        except Exception:  # noqa: BLE001 — no nvidia-smi on this host: utilisation is simply not reported
            return
        await asyncio.sleep(1.0)


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
    samples: list[dict[int, float]] = []
    stop_ev = asyncio.Event()
    sampler = asyncio.create_task(sample_gpus(samples, stop_ev))
    stop = time.perf_counter() + seconds
    counter = iter(range(10**9))
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=users + 4)) as client:
        async def user():
            while time.perf_counter() < stop:
                recs.append(await one_request(client, url, questions[next(counter) % len(questions)]))

        t0 = time.perf_counter()
        await asyncio.gather(*(user() for _ in range(users)))
        wall = time.perf_counter() - t0
    stop_ev.set()
    await sampler
    return summarize_load(recs, wall, users) | mean_by_gpu(samples)


def main() -> None:
    from corpus import QASet

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--users", default="1,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--n", type=int, default=100, help="dev questions to cycle")
    ap.add_argument("--tag", default="", help="label stored with every line (e.g. the GPU layout)")
    ap.add_argument("--out", default="")
    ap.add_argument("--find-max", action="store_true", help="search the largest user count that meets the SLO instead of sweeping --users")
    ap.add_argument("--slo-p95", type=float, default=10.0, help="--find-max: p95 total latency bound in seconds")
    ap.add_argument("--limit", type=int, default=256, help="--find-max: upper bound on users")
    a = ap.parse_args()
    qs = [q.text for q in QASet().load("dev")[: a.n]]
    if a.find_max:
        def measure(u):
            row = {"tag": a.tag} | asyncio.run(run_level(a.url, qs, u, a.seconds))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
            return row

        best, _ = find_max_users(measure, start=int(a.users.split(",")[0]), limit=a.limit, slo_p95=a.slo_p95)
        print(json.dumps({"tag": a.tag, "max_users_at_slo": best, "slo_p95_s": a.slo_p95}), flush=True)
        return
    for u in (int(x) for x in a.users.split(",")):
        row = {"tag": a.tag} | asyncio.run(run_level(a.url, qs, u, a.seconds))
        print(json.dumps(row), flush=True)
        problem = level_problem(row)
        if problem:
            print(f"PROBLEM at users={u}: {problem}", flush=True)
            raise SystemExit(2)   # fail fast: a broken stack must not produce a full table of zeros
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
