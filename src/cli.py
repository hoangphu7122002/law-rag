"""Command line. Every `make` target calls one subcommand here.

    python -m cli data                                   # build data/processed from data/raw
    python -m cli run configs/exp/B0_baseline.yaml [--backend l40s] [--set retrieve.k=10 ...] [--no-score]
    python -m cli score data/runs/<run>                       # (re)score a run dir
    python -m cli compare data/runs/<a> data/runs/<b> [--metric rouge_l]
    python -m cli box <action> [...]                     # remote GPU box (src/ops/box.py), see `cli box -h`
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    sub.add_parser("vbpl", help="extra corpus from vbpl.vn (HF th1nhng0/vietnamese-legal-documents)")
    r = sub.add_parser("run")
    r.add_argument("exp")
    r.add_argument("--backend")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--no-score", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--no-cosine", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--metric", default="rouge_l")
    b = sub.add_parser("box", help="remote GPU box: push / setup / serve / run / status / stop / pull / tunnel")
    b.add_argument("action", choices=["push", "push-data", "setup", "serve", "run", "status", "stop", "pull", "tunnel",
                                      "watch", "autosync"])
    b.add_argument("target", nargs="?", help="serve: model id · run: exp yaml · stop: session (llm-g0, embed-g0, run-<exp>) · autosync: on|off")
    b.add_argument("--gpu", type=int, default=0)
    b.add_argument("--kind", default="llm", choices=["llm", "embed", "rerank"], help="serve: which vLLM server")
    b.add_argument("--on", default="l40s", help="backend file with a `box:` block")
    args, extra = ap.parse_known_args()
    if extra and args.cmd != "box":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.cmd == "data":
        from corpus import DatasetBuilder

        DatasetBuilder().build()
    elif args.cmd == "vbpl":
        from corpus.vbpl import VbplImporter

        VbplImporter().build()
    elif args.cmd == "run":
        from config import load
        from pipeline import Experiment

        summary = Experiment(load(args.exp, args.set, args.backend)).execute(score=not args.no_score)
        print(json.dumps(summary, indent=1) if summary else "answers written; score later with `cli score`")
    elif args.cmd == "score":
        from evaluate import Evaluator
        from runs import Run

        print(json.dumps(Evaluator(cosine=not args.no_cosine).score(Run(Path(args.run))), indent=1))
    elif args.cmd == "compare":
        from evaluate import PairedBootstrap
        from runs import Run

        a, b = (({r["qid"]: r[args.metric] for r in Run(Path(p)).read_json("metrics.json")["rows"]}) for p in (args.a, args.b))
        print(json.dumps(PairedBootstrap().compare(a, b), indent=1))

    elif args.cmd == "box":
        from ops import RemoteBox

        box = RemoteBox.from_backend(args.on)
        needs_target = {"serve", "run", "stop", "autosync"}
        if args.action in needs_target and not args.target:
            ap.error(f"box {args.action} needs a target")
        {
            "push": box.push_code, "push-data": box.push_data, "setup": box.setup, "status": box.status,
            "pull": box.pull, "tunnel": box.tunnel,
            "serve": lambda: box.serve(args.target, args.gpu, args.kind),
            "run": lambda: box.run(args.target, args.gpu, extra),
            "stop": lambda: box.stop(args.target),
            "watch": box.watch,
            "autosync": lambda: box.autosync(args.target == "on"),
        }[args.action]()


if __name__ == "__main__":
    main()
