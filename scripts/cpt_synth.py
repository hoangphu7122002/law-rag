"""CPT augmentation: paraphrases / forward+reverse QA / summaries per law article via DeepSeek (vibe_call.py). See src/train/cpt_synth.py.

    PYTHONPATH=src python3 scripts/cpt_synth.py --articles data/cpt/articles.jsonl --dry-run
    PYTHONPATH=src python3 scripts/cpt_synth.py --articles data/cpt/articles.jsonl --limit-articles 200 --out data/cpt/aug_pilot.jsonl
Resumable (finished batch ids skipped). Dev/test gold articles are never sent to the teacher.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from train import synth_gen as sg
from train.cpt_synth import Article, batches, make_prompt, process


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/cpt/articles.jsonl")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/cpt/aug.jsonl")
    ap.add_argument("--limit-articles", type=int, default=0)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-tokens", type=int, default=60_000_000)
    ap.add_argument("--calls-per-minute", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vibe", default=str(Path("~/.claude/skills/vibe/vibe_call.py").expanduser()))
    ap.add_argument("--model", default="deepseekflash")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects, done_f, pdir = out.with_suffix(".rejects.jsonl"), out.with_suffix(".done"), out.with_suffix(".prompts")
    excluded, _, blocked = sg.load_eval_filters(a.processed)

    def norm_key(k: str) -> str:
        law, _, d = k.partition("#")
        return f"{law.upper()}#{d.lower()}"
    arts = []
    for line in Path(a.articles).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if norm_key(r["key"]) in excluded or not a.min_chars <= len(r["text"]) <= a.max_chars:
            continue
        arts.append(Article(r["key"], r["title"], r["text"]))
    random.Random(a.seed).shuffle(arts)
    arts = arts[: a.limit_articles or None]
    bs = batches(arts, a.per_call)
    done = sg.read_done(done_f)
    todo = [b for b in bs if b.id not in done]
    prompts = {b.id: make_prompt(b) for b in todo}
    est = sum(sg.estimate_tokens(prompts[b.id], len(b.items)) for b in todo)
    print(f"articles {len(arts)} (dev/test gold excluded: {len(excluded)}); batches {len(bs)}, done {len(done)}, todo {len(todo)}; est tokens {est:,} (cap {a.max_tokens:,})")
    if a.dry_run:
        return

    pdir.mkdir(exist_ok=True)
    used = calls = n_ok = 0
    last = 0.0
    for b in todo:
        add = sg.estimate_tokens(prompts[b.id], len(b.items))
        if (a.max_calls and calls >= a.max_calls) or not sg.within_budget(used, add, a.max_tokens):
            print("budget reached, stop")
            break
        wait = last + sg.min_interval(a.calls_per_minute) - time.time()
        if wait > 0:
            time.sleep(wait)
        pf = pdir / f"{b.id}.txt"
        pf.write_text(prompts[b.id])
        last = time.time()
        r = subprocess.run([sys.executable, a.vibe, "--model", a.model, "--thinking", "off", "--label", "cpt-synth",
                            "--prompt-file", str(pf)], capture_output=True, text=True)
        calls, used = calls + 1, used + add
        if r.returncode != 0:
            print(f"{b.id}: call failed rc={r.returncode} {r.stderr[-200:]!r}", file=sys.stderr)
            continue
        rows, rej = process(r.stdout, b, blocked)
        with open(out, "a") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        with open(rejects, "a") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rej)
        sg.mark_done(done_f, b.id)
        n_ok += len(rows)
        print(f"{b.id}: ok {len(rows)} rej {len(rej)} (calls {calls}, ~tokens {used:,})")
    print(f"finished: {n_ok} rows this run -> {out}")


if __name__ == "__main__":
    main()
