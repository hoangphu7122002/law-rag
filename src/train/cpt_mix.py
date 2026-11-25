"""Mix the CPT sources into one packed training file (docs/research: paraphrase/QA/summary forms of each fact, plus replay against forgetting).

Sources (token budgets are per epoch): raw laws, DeepSeek augmentation (paraphrase / QA / summary), external open-licence laws, general Vietnamese
replay (fineweb-2), SFT rows replay. Every document gets an EOS, documents of all sources are shuffled together, then cut into seq_len sequences.

    PYTHONPATH=src python -m train.cpt_mix --out data/cpt/mix.jsonl
"""
import argparse
import json
import random
from pathlib import Path


def take(docs: list[list[int]], budget: int, seed: int) -> list[list[int]]:
    """Random documents until the token budget is reached (the last one may overshoot by its length)."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    out, n = [], 0
    for i in order:
        if n >= budget:
            break
        out.append(docs[i])
        n += len(docs[i])
    return out


def mix(sources: dict[str, list[list[int]]], budgets: dict[str, int], seq_len: int, eos: int, seed: int):
    """-> (sequences of exactly seq_len, {source: tokens taken})."""
    from train.cpt_data import pack

    docs, stats = [], {}
    for k, ds in sources.items():
        chosen = take(ds, budgets[k], seed)
        stats[k] = sum(map(len, chosen))
        docs += chosen
    random.Random(seed).shuffle(docs)
    return pack(docs, seq_len, eos), stats


def _lines(path):
    return (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--raw", default="data/cpt/laws.jsonl", help="packed ids of the raw laws (train.cpt_data)")
    ap.add_argument("--aug", default="data/cpt/aug_dedup.jsonl")
    ap.add_argument("--ext", default="data/cpt/ext.jsonl")
    ap.add_argument("--sft", default="data/train/q35_D3/train.jsonl")
    ap.add_argument("--budget", default="raw=8e6,aug=9e6,ext_law=6e6,ext_general=4e6,sft=1.5e6")
    ap.add_argument("--out", default="data/cpt/mix.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    eos = tok.eos_token_id
    enc = lambda texts: tok(list(texts), add_special_tokens=False)["input_ids"]   # noqa: E731
    ext = list(_lines(a.ext))
    sources = {
        "raw": [s for s in (r["input_ids"] for r in _lines(a.raw))],   # already packed 2048-token sequences: kept whole
        "aug": enc(r["text"] for r in _lines(a.aug)),
        "ext_law": enc(r["text"] for r in ext if r["source"] != "fineweb2_vie"),
        "ext_general": enc(r["text"] for r in ext if r["source"] == "fineweb2_vie"),
        "sft": enc(r["prompt"] + r["completion"] for r in _lines(a.sft)),
    }
    budgets = {k: int(float(v)) for k, v in (x.split("=") for x in a.budget.split(","))}
    seqs, stats = mix(sources, budgets, a.seq_len, eos, a.seed)
    Path(a.out).write_text("".join(json.dumps({"input_ids": s}) + "\n" for s in seqs))
    print(f"{len(seqs)} sequences ({len(seqs) * a.seq_len / 1e6:.1f}M tokens) -> {a.out}; per source {stats}; available { {k: sum(map(len, v)) for k, v in sources.items()} }")


if __name__ == "__main__":
    main()
