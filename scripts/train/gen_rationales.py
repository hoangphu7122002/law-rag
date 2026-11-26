"""Thinking-SFT data: rationales from a served stronger model (Qwen3.5-9B, thinking off) for every train/val row of a data dir; invalid rationales (leaks, wrong Điều, bad length)
are retried once, then the row is dropped. Writes <out>/train.jsonl and val.jsonl in the thinking format (see train.synth.rationale).
    PYTHONPATH=src python scripts/train/gen_rationales.py data/train/q35_C data/train/q35_T --url http://localhost:8001/v1 --model Qwen/Qwen3.5-9B"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from train.synth.rationale import assemble, rationale_request, valid_rationale

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--conc", type=int, default=24)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()


def ask(row: dict) -> str | None:
    ref = row["completion"]
    for temp in (0.3, 0.7):
        try:
            r = requests.post(f"{a.url}/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": rationale_request(row["prompt"], ref)}],
                                                                  "temperature": temp, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600).json()
            text = r["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
        if valid_rationale(text, ref):
            return text
    return None


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    rows = [json.loads(line) for line in open(Path(a.src) / f"{split}.jsonl", encoding="utf-8")]
    rows = rows[:a.limit] if a.limit else rows
    with ThreadPoolExecutor(a.conc) as ex:
        rats = list(ex.map(ask, rows))
    kept = [assemble(r, t) for r, t in zip(rows, rats) if t]
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    print(json.dumps({"split": split, "rows": len(rows), "kept": len(kept)}), flush=True)
