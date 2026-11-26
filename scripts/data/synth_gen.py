"""Synthetic Q/A generation with a teacher LLM (DeepSeek-V4 flash via vibe_call.py). See src/train/synth_gen.py.

    PYTHONPATH=src python3 scripts/synth_gen.py --dry-run
    PYTHONPATH=src python3 scripts/synth_gen.py --n-items 2000 --max-calls 50   # resumable: finished batch ids are skipped
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from train import synth_gen as sg


def parse_mix(s: str) -> dict:
    return {k: float(v) for k, v in (x.split("=") for x in s.split(",") if x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--dirs", nargs="*", default=["corpus", "vbpl_cited", "web"], help="subdirs of --processed")
    ap.add_argument("--out", default="data/synth/synth_gen.jsonl")
    ap.add_argument("--n-items", type=int, default=2000)
    ap.add_argument("--mix", default=",".join(f"{k}={v}" for k, v in sg.DEFAULT_MIX.items()))
    ap.add_argument("--per-law", type=int, default=5)
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="only the first N batches of the plan")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    ap.add_argument("--calls-per-minute", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vibe", default=str(Path("~/.claude/skills/vibe/vibe_call.py").expanduser()))
    ap.add_argument("--model", default="deepseekflash")
    ap.add_argument("--focus-train-gold", action="store_true", help="emphasise the gold laws of the TRAIN questions (target domain)")
    ap.add_argument("--per-law-focus", type=int, default=15)
    ap.add_argument("--other-frac", type=float, default=0.3, help="with --focus-train-gold: share of the other laws that stay in the pool")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects, done_f, pdir = out.with_suffix(".rejects.jsonl"), out.with_suffix(".done"), out.with_suffix(".prompts")
    excluded, eval_q, blocked = sg.load_eval_filters(a.processed)
    focus = None
    if a.focus_train_gold:   # laws cited by TRAIN references only (dev / test are never used for sampling)
        import pandas as pd

        sp = json.loads((Path(a.processed) / "split.json").read_text())
        g = pd.read_parquet(Path(a.processed) / "gold.parquet")
        focus = {k for ds in g[g.qid.isin(set(sp["train"]))].docs for k in ds}
    pool = sg.build_pool([Path(a.processed) / d for d in a.dirs], excluded, a.per_law, a.seed, focus, a.per_law_focus,
                         a.other_frac if focus else 1.0)
    batches = sg.plan(pool, a.n_items, parse_mix(a.mix), a.seed, a.per_call)[: a.limit]
    done = sg.read_done(done_f)
    todo = [b for b in batches if b.id not in done]
    prompts = {b.id: sg.make_prompt(b) for b in todo}
    est = sum(sg.estimate_tokens(prompts[b.id], len(b.items)) for b in todo)
    by_pat = {p: sum(1 for b in todo if b.pattern == p) for p in sg.PATTERNS}
    print(f"pool {len(pool)} articles ({len(excluded)} dev/test gold excluded); batches {len(batches)}, done {len(done)}, "
          f"todo {len(todo)} {by_pat}; est tokens {est:,} (cap {a.max_tokens:,})")
    if a.dry_run:
        return

    pdir.mkdir(exist_ok=True)
    seen = {sg.norm(json.loads(l)["question"]).lower() for l in out.read_text().splitlines()} if out.exists() else set()
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
        r = subprocess.run([sys.executable, a.vibe, "--model", a.model, "--thinking", "off", "--label", "tdata-synth",
                            "--prompt-file", str(pf)], capture_output=True, text=True)
        calls, used = calls + 1, used + add
        if r.returncode != 0:
            print(f"{b.id}: call failed rc={r.returncode} {r.stderr[-200:]!r}", file=sys.stderr)
            continue
        rows, rej = sg.process(r.stdout, b, eval_q, blocked, seen)
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
