"""Teacher repair of wrong ReAct trajectories (DeepSeek flash via vibe_call.py).
Usage: PYTHONPATH=src python3 scripts/train/teacher_repair.py --traj RUN/trajectories.jsonl --out data/train/repair.jsonl [--tools-url URL] [--dry-run] [--limit N]
Rejected repairs go to <out>.rejected.jsonl; rerun resumes (skips qids already in either file)."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

from train.react.repair_run import done_qids, plan, run_repairs, select_items

VIBE = os.path.expanduser("~/.claude/skills/vibe/vibe_call.py")


def vibe_teacher(prompt: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(prompt)
        path = f.name
    try:
        r = subprocess.run([sys.executable, VIBE, "--model", "deepseekflash", "--thinking", "off", "--label", "tdata-repair", "--prompt-file", path],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"vibe_call exit {r.returncode}")
        return r.stdout
    finally:
        os.unlink(path)


def http_tool(url: str, exp: str = "configs/exp/P1_9b.yaml"):
    """Tools through the shared tools endpoint (generate/tools_service.py); results rendered exactly as the agent sees them (ReactGenerator.run_tool)."""
    from config import load
    from generate.react import ReactGenerator
    from generate.tools_service import RemoteTools

    cfg = load(exp, ["generate.agent_tools=[search_law,get_article,kg_related]", "generate.search_with_question=true"])
    gen = ReactGenerator(cfg.generate, cfg.backend, RemoteTools(url))

    def run_tool(name, args):
        return gen.run_tool(name, args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
    return run_tool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa", default="data/processed/qa.parquet")
    ap.add_argument("--gold", default="data/processed/gold.parquet")
    ap.add_argument("--tools-url", default=None, help="tools server base URL (POST /<tool> with JSON args)")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--calls-per-min", type=float, default=20)
    ap.add_argument("--max-tokens", type=int, default=60_000_000)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    qa = {r["qid"]: r for r in pl.read_parquet(a.qa).iter_rows(named=True)}
    gold = {r["qid"]: set(r["pairs"] or ()) for r in pl.read_parquet(a.gold).iter_rows(named=True)}
    rows = [json.loads(l) for l in Path(a.traj).read_text().splitlines() if l.strip()]
    rej = a.out + ".rejected.jsonl"
    items = select_items(rows, qa, gold, done=done_qids(a.out, rej), limit=a.limit)
    p = plan(items, a.batch_size)
    print(f"items {p['items']} batches {p['batches']} est_tokens {p['est_tokens']}")
    if a.dry_run:
        return
    if not a.tools_url:
        sys.exit("--tools-url required")
    s = run_repairs(items, vibe_teacher, http_tool(a.tools_url), a.out, rej, a.batch_size, a.calls_per_min, a.max_tokens, a.max_calls)
    print(s)


if __name__ == "__main__":
    main()
