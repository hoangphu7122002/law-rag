"""Sample candidates for DPO: per train row (src/train/sft/sft_data.py format) one greedy + (n-1) samples at T0.7 from a vLLM endpoint.

    python scripts/train/pref_sample.py --data data/train/q35_v3b --out data/train/pref_v3b/cands.jsonl --base-url http://localhost:8001/v1 --model <served name>
Output rows: {"qid", "prompt", "reference", "cands"}. Rows whose completion is a refusal are skipped. Resumable (appends, skips done qids).
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.data) / "train.jsonl")]
    from train.preference.grpo import is_refusal
    rows = [r for r in rows if not is_refusal(r["completion"].replace("<|im_end|>", ""))]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["qid"] not in done]
    print(f"{len(rows)} rows, {len(todo)} to sample", flush=True)

    def one(r: dict) -> dict:
        base = {"model": a.model, "prompt": r["prompt"], "max_tokens": 1024, "top_p": 0.95}
        g = post(f"{a.base_url}/completions", {**base, "temperature": 0.0})["choices"][0]["text"]
        s = post(f"{a.base_url}/completions", {**base, "temperature": 0.7, "n": a.n - 1})["choices"]
        return {"qid": r["qid"], "prompt": r["prompt"], "reference": r["completion"].replace("<|im_end|>", ""),
                "cands": [g] + [c["text"] for c in s]}

    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        for i, row in enumerate(ex.map(one, todo)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main()
