"""Thinking-SFT data from GENUINE reasoning: Qwen3.5-9B (served with thinking on, the prompt opens <think>) answers each train/val prompt of a data dir WITHOUT seeing the reference,
n samples per row; a sample is kept when its final answer cites exactly the reference's Điều numbers and overlaps it (ROUGE-L >= min), i.e. the reasoning found the right provision.
The training row = prompt (opens <think>) + the kept thought + </think> + the REFERENCE answer (our register), best sample by ROUGE-L, shortest thought on ties.
    PYTHONPATH=src python scripts/train/gen_thinking.py data/train/q35_C data/train/q35_T --url http://localhost:8001/v1 --model Qwen/Qwen3.5-9B --n 4"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from train.synth.rationale import assemble, pick_thought, rationale_request, split_think, to_think_prompt, valid_rationale

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
ap.add_argument("--no-rationalise", action="store_true", help="skip pass 2 (9B writes a short rationale, shown the reference, for rows pass 1 could not solve)")
a = ap.parse_args()


def sample(row: dict) -> dict | None:
    prompt = to_think_prompt(row["prompt"])
    try:
        r = requests.post(f"{a.url}/completions", json={"model": a.model, "prompt": prompt, "n": a.n, "temperature": 0.7, "top_p": 0.95, "max_tokens": a.max_tokens,
                                                         "skip_special_tokens": False}, timeout=1800).json()
        outs = [c["text"].replace("<|im_end|>", "") for c in r["choices"]]
    except Exception:
        return None
    cands = [parts for text in outs if (parts := split_think(text))]
    thought = pick_thought(cands, row["completion"], a.min_rouge)
    if thought is not None:
        return {"qid": row["qid"], "prompt": prompt, "completion": thought + "\n</think>\n\n" + row["completion"], "source": "genuine"}
    return None


def rationalise(row: dict) -> dict | None:
    """Pass 2 (STaR rationalisation by our own 9B): rows the blind pass could not solve are exactly the hard tail/mid ones, so write a short reasoning shown the reference."""
    for _ in range(3):
        try:
            r = requests.post(f"{a.url}/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": rationale_request(row["prompt"], row["completion"])}],
                                                                  "temperature": 0.7, "max_tokens": 600, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600).json()
            text = r["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
        if valid_rationale(text, row["completion"]):
            return {**assemble(row, text), "source": "rationalised"}
    return None


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    rows = [json.loads(line) for line in open(Path(a.src) / f"{split}.jsonl", encoding="utf-8")]
    rows = rows[:a.limit] if a.limit else rows
    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(sample, rows))
    if not a.no_rationalise:
        todo = [row for row, r in zip(rows, res) if r is None]
        with ThreadPoolExecutor(a.conc) as ex:
            res2 = list(ex.map(rationalise, todo))
        it = iter(res2)
        res = [r if r else next(it) for r in res]
    kept = [r for r in res if r]
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k != "source"}, ensure_ascii=False) + "\n" for r in kept)   # SFT rows stay {qid, prompt, completion}
    (out / f"{split}_sources.json").write_text(json.dumps({r["qid"]: r["source"] for r in kept}))
    print(json.dumps({"split": split, "rows": len(rows), "kept": len(kept), "genuine": sum(r["source"] == "genuine" for r in kept), "rationalised": sum(r["source"] == "rationalised" for r in kept)}), flush=True)
