"""CPT augmentation: paraphrases / forward+reverse QA / summaries per law article via DeepSeek (vibe_call.py). See src/train/cpt_synth.py.

    PYTHONPATH=src python3 scripts/data/cpt_synth.py --articles data/cpt/articles.jsonl --dry-run
    PYTHONPATH=src python3 scripts/data/cpt_synth.py --articles data/cpt/articles.jsonl --limit-articles 200 --out data/cpt/aug_pilot.jsonl
Resumable (finished batch ids skipped). Dev/test gold articles are never sent to the teacher.
"""
import argparse
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train import synth_gen as sg
from train.cpt_synth import Article, batches, groups, make_prompt, make_prompt_multi, process, process_multi


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
    ap.add_argument("--mode", choices=["single", "multi"], default="single", help="multi: groups of consecutive articles of one law -> multi-article QA, scenarios, group summary")
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
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
    if a.mode == "multi":
        units = groups(arts, a.group_size)
        random.Random(a.seed).shuffle(units)
        units = units[: a.limit_articles or None]
        bs = batches(units, a.per_call, prefix="m")
        make, handle = make_prompt_multi, process_multi
        arts = units
    else:
        random.Random(a.seed).shuffle(arts)
        arts = arts[: a.limit_articles or None]
        bs = batches(arts, a.per_call)
        make, handle = make_prompt, process
    done = sg.read_done(done_f)
    todo = [b for b in bs if b.id not in done]
    prompts = {b.id: make(b) for b in todo}
    est = sum(sg.estimate_tokens(prompts[b.id], len(b.items)) for b in todo)
    print(f"{a.mode} units {len(arts)} (dev/test gold excluded: {len(excluded)}); batches {len(bs)}, done {len(done)}, todo {len(todo)}; est tokens {est:,} (cap {a.max_tokens:,})")
    if a.dry_run:
        return

    pdir.mkdir(exist_ok=True)
    lock = threading.Lock()
    stats = {"calls": 0, "ok": 0, "used": 0}
    gate = {"last": 0.0}

    def work(b):
        add = sg.estimate_tokens(prompts[b.id], len(b.items))
        with lock:
            if (a.max_calls and stats["calls"] >= a.max_calls) or not sg.within_budget(stats["used"], add, a.max_tokens):
                return
            stats["calls"] += 1
            stats["used"] += add
            wait = gate["last"] + sg.min_interval(a.calls_per_minute) - time.time()
            gate["last"] = max(time.time(), gate["last"] + sg.min_interval(a.calls_per_minute))
        if wait > 0:
            time.sleep(wait)
        pf = pdir / f"{b.id}.txt"
        pf.write_text(prompts[b.id])
        r = subprocess.run([sys.executable, a.vibe, "--model", a.model, "--thinking", "off", "--label", "cpt-synth",
                            "--prompt-file", str(pf)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{b.id}: call failed rc={r.returncode} {r.stderr[-200:]!r}", file=sys.stderr, flush=True)
            return
        rows, rej = handle(r.stdout, b, blocked)
        with lock:
            with open(out, "a") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
            with open(rejects, "a") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rej)
            sg.mark_done(done_f, b.id)
            stats["ok"] += len(rows)
            print(f"{b.id}: ok {len(rows)} rej {len(rej)} (calls {stats['calls']}, ~tokens {stats['used']:,})", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    n_ok, used = stats["ok"], stats["used"]
    print(f"finished: {n_ok} rows this run -> {out}")


if __name__ == "__main__":
    main()
