"""Mix the CPT sources into one packed training file (docs/research: paraphrase/QA/summary forms of each fact, plus replay against forgetting).

Sources (token budgets are per epoch): raw laws, DeepSeek augmentation (paraphrase / QA / summary), external open-licence laws, general Vietnamese
replay (fineweb-2), SFT rows replay. Every document gets an EOS, documents of all sources are shuffled together, then cut into seq_len sequences.

    PYTHONPATH=src python -m train.cpt.cpt_mix --out data/cpt/mix.jsonl
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
    from train.cpt.cpt_data import pack

    docs, stats = [], {}
    for k, ds in sources.items():
        chosen = take(ds, budgets[k], seed)
        stats[k] = sum(map(len, chosen))
        docs += chosen
    random.Random(seed).shuffle(docs)
    return pack(docs, seq_len, eos), stats


def mix_np(sources: dict, budgets: dict, seq_len: int, eos: int, seed: int):
    """Like mix() for numpy uint32 documents (100M+ tokens do not fit as python lists) -> (matrix [n, seq_len] uint32, {source: tokens})."""
    import numpy as np

    docs, stats = [], {}
    for k, ds in sources.items():
        chosen = take(ds, budgets[k], seed)
        stats[k] = int(sum(len(d) for d in chosen))
        docs += chosen
    random.Random(seed).shuffle(docs)
    sep = np.array([eos], dtype=np.uint32)
    flat = np.concatenate([x for d in docs for x in (d, sep)]) if docs else np.zeros(0, dtype=np.uint32)
    n = len(flat) // seq_len
    return flat[: n * seq_len].reshape(n, seq_len), stats


class NpyDataset:
    """Map-style dataset over a [n, seq_len] uint32 .npy matrix (memory-mapped): input_ids = labels = the row. start/stop select a slice."""

    def __init__(self, path: str, start: int = 0, stop: int | None = None):
        import numpy as np

        self.m = np.load(path, mmap_mode="r")
        self.start, self.stop = start, self.m.shape[0] if stop is None else stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, i):
        import numpy as np
        import torch

        ids = torch.from_numpy(np.asarray(self.m[self.start + i], dtype=np.int64))
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def _lines(path):
    return (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--raw", default="data/cpt/laws.jsonl", help="packed ids of the raw laws (train.cpt.cpt_data)")
    ap.add_argument("--aug", nargs="+", default=["data/cpt/aug_dedup.jsonl", "data/cpt/aug_multi.jsonl"])
    ap.add_argument("--ext", default="data/cpt/ext_v3.jsonl")
    ap.add_argument("--sft", default="data/train/q35_D3/train.jsonl")
    ap.add_argument("--budget", default="raw=9e6,aug=1e9,ext_law=1e9,ext_general=12e6,sft=8e6", help="tokens; 1e9 = take everything")
    ap.add_argument("--out", default="data/cpt/mix.npy")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)

    def enc(texts, bs=2000):
        out, buf = [], []
        for t in texts:
            buf.append(t)
            if len(buf) == bs:
                out += [np.array(x, dtype=np.uint32) for x in tok(buf, add_special_tokens=False)["input_ids"]]
                buf = []
        if buf:
            out += [np.array(x, dtype=np.uint32) for x in tok(buf, add_special_tokens=False)["input_ids"]]
        return out

    general = {"wikipedia", "fineweb"}
    ext_law, ext_gen = [], []
    for r in _lines(a.ext):
        (ext_gen if r["source"] in general else ext_law).append(r["text"])
    sources = {
        "raw": [np.array(r["input_ids"], dtype=np.uint32) for r in _lines(a.raw)],   # already packed 2048-token sequences: kept whole
        "aug": enc(r["text"] for p in a.aug if Path(p).exists() for r in _lines(p)),
        "ext_law": enc(ext_law),
        "ext_general": enc(ext_gen),
        "sft": enc(r["prompt"] + r["completion"] for r in _lines(a.sft)),
    }
    budgets = {k: int(float(v)) for k, v in (x.split("=") for x in a.budget.split(","))}
    arr, stats = mix_np(sources, budgets, a.seq_len, tok.eos_token_id, a.seed)
    np.save(a.out, arr)
    avail = {k: int(sum(len(d) for d in v)) for k, v in sources.items()}
    print(f"{len(arr)} sequences ({arr.size / 1e6:.1f}M tokens) -> {a.out}; per source {stats}; available {avail}")


if __name__ == "__main__":
    main()
