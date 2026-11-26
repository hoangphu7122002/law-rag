"""Persistent Python kernel for the weight-space agent (one per search). The model stays loaded; each submitted snippet counts as one analysis (<=300).
    start (box, tmux):  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/train/wsp_kernel.py --base data/models/dpo_Vp_A2 --feedback data/wsp/feedback.jsonl --work data/wsp/run1
    agent:              echo 'print(env.generate(range(8), weights="original")["original"]["rouge_l"])' | PYTHONPATH=src .venv/bin/python scripts/train/wsp_kernel.py --work data/wsp/run1 --submit
Namespace: env, torch, json, math, statistics, np. Variables persist between snippets."""
import argparse
import contextlib
import io
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--base")
ap.add_argument("--feedback")
ap.add_argument("--submit", action="store_true")
a = ap.parse_args()
work = Path(a.work)
inbox, outbox = work / "in", work / "out"
inbox.mkdir(parents=True, exist_ok=True)
outbox.mkdir(parents=True, exist_ok=True)

if a.submit:
    import sys
    n = len(list(inbox.glob("*.py")))
    (inbox / f"{n:04d}.py.tmp").write_text(sys.stdin.read())
    (inbox / f"{n:04d}.py.tmp").rename(inbox / f"{n:04d}.py")
    while not (outbox / f"{n:04d}.txt").exists():
        time.sleep(1)
    print((outbox / f"{n:04d}.txt").read_text())
    raise SystemExit

import json, math, statistics  # noqa: E401
import numpy as np
import torch

from train.weights.wsp_env import Env

env = Env(a.base, a.feedback, a.work)
ns = {"env": env, "torch": torch, "json": json, "math": math, "statistics": statistics, "np": np}
print("kernel ready", flush=True)
while True:
    for f in sorted(inbox.glob("*.py")):
        o = outbox / (f.stem + ".txt")
        if o.exists():
            continue
        buf = io.StringIO()
        try:
            left = env.analysis()
            with contextlib.redirect_stdout(buf):
                exec(compile(f.read_text(), str(f), "exec"), ns)
            text = buf.getvalue() + f"\n[analyses left: {left}]"
        except Exception:
            text = buf.getvalue() + traceback.format_exc()[-4000:]
        o.write_text(text)
    time.sleep(0.5)
