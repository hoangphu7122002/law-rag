"""Thinking-SFT data from GENUINE reasoning: Qwen3.5-9B (served with thinking on, the prompt opens <think>) answers each train/val prompt of a data dir WITHOUT seeing the reference,
n samples per row; a sample is kept when its final answer cites exactly the reference's Điều numbers and overlaps it (ROUGE-L >= min), i.e. the reasoning found the right provision.
The training row = prompt (opens <think>) + the kept thought + </think> + the REFERENCE answer (our register), best sample by ROUGE-L, shortest thought on ties.
    PYTHONPATH=src python scripts/train/gen_thinking.py data/train/q35_C data/train/q35_T --url http://localhost:8001/v1 --model Qwen/Qwen3.5-9B --n 4"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from evaluate.metrics import rouge_l_score
from train.synth.rationale import accept_trace, split_think, to_think_prompt

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--conc", type=int, default=48)
ap.add_argument("--max-tokens", type=int, default=2200)
ap.add_argument("--min-rouge", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()


def sample(row: dict) -> dict | None:
    prompt = to_think_prompt(row["prompt"])
    try:
        r = requests.post(f"{a.url}/completions", json={"model": a.model, "prompt": prompt, "n": a.n, "temperature": 0.7, "top_p": 0.95, "max_tokens": a.max_tokens,
                                                         "skip_special_tokens": False}, timeout=1800).json()
        outs = [c["text"].replace("<|im_end|>", "") for c in r["choices"]]
    except Exception:
        return None
    best = None
    for text in outs:
        parts = split_think(text)
        if not parts or not accept_trace(parts[0], parts[1], row["completion"], a.min_rouge):
            continue
        key = (rouge_l_score(parts[1], row["completion"]), -len(parts[0]))
        if best is None or key > best[0]:
            best = (key, parts[0])
    return None if best is None else {"qid": row["qid"], "prompt": prompt, "completion": best[1] + "\n</think>\n\n" + row["completion"]}


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    rows = [json.loads(line) for line in open(Path(a.src) / f"{split}.jsonl", encoding="utf-8")]
    rows = rows[:a.limit] if a.limit else rows
    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(sample, rows))
    kept = [r for r in res if r]
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    print(json.dumps({"split": split, "rows": len(rows), "kept": len(kept)}), flush=True)
