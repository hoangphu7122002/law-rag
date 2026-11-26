"""Closed-book cloze probe for the CPT go/no-go (docs/research): does the model complete a number of a law article without any context?

Item = law title + "Điều N <heading>" + the article text up to a number (in a clause body, not the heading), target = that number. A format none of the CPT
tasks used (paraphrase / QA / summary / raw text), so a gain means the fact was learned, not a template. Report groups separately: articles the CPT
saw in augmented form (train) vs. gold articles of dev/test (raw text only, no augmentation).

    PYTHONPATH=src python -m train.cpt_probe --model data/models/q35_cpt_lora --base Qwen/Qwen3.5-2B --articles data/cpt/articles.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def make_items(articles: list[dict], per_article: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for a in articles:
        head, _, body = a["text"].partition("\n")
        cands = [m for m in NUM.finditer(body) if m.start() > 40 and body[m.start() - 1] != "\n"]   # not a clause marker
        rng.shuffle(cands)
        for m in cands[:per_article]:
            prompt = f'{a["title"]}\n{head}\n{body[:m.start()]}'.rstrip()
            items.append({"key": a["key"], "prompt": prompt, "target": m.group()})
    return items


def score(items: list[dict], preds: list[str]) -> float:
    hit = sum(1 for it, p in zip(items, preds) if re.match(rf"\s*{re.escape(it['target'])}(?![\d])", p))
    return hit / max(len(items), 1)


def generate(model_path: str, base: str, prompts: list[str], max_new: int = 6, bs: int = 16) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=1536).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--articles", default="data/cpt/articles.jsonl")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/cpt/probe.json")
    a = ap.parse_args()
    from train import synth_gen as sg

    excluded, _, _ = sg.load_eval_filters(a.processed)
    norm = lambda k: f"{k.split('#')[0].upper()}#{k.split('#')[1].lower()}"   # noqa: E731
    arts = [json.loads(l) for l in Path(a.articles).read_text(encoding="utf-8").splitlines()]
    train = [x for x in arts if norm(x["key"]) not in excluded and 150 <= len(x["text"]) <= 6000]
    held = [x for x in arts if norm(x["key"]) in excluded]
    rng = random.Random(a.seed)
    rng.shuffle(train), rng.shuffle(held)
    groups = {"train_articles": make_items(train[: a.n], 1, a.seed)[: a.n], "devtest_gold_raw_only": make_items(held, 1, a.seed)[: a.n]}
    res = {}
    for m in a.models:
        res[m] = {}
        for g, items in groups.items():
            res[m][g] = {"n": len(items), "cloze_acc": round(score(items, generate(m, a.base, [i["prompt"] for i in items])), 4)}
        print(m, res[m], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
