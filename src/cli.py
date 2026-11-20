"""Command line. Every `make` target calls one subcommand here.

    python -m cli data                                   # build data/processed from data/raw
    python -m cli run configs/exp/B0_baseline.yaml [--backend l40s] [--set retrieve.k=10 ...] [--no-score]
    python -m cli run <exp.yaml> --retrieval-only --set retrieve.k=20  # doc@k / Điều@k only, no LLM
    python -m cli score data/runs/<run>                       # (re)score a run dir
    python -m cli retrieval data/runs/<run>                    # doc@k / Điều@k of the retrieved passages
    python -m cli sft-data configs/exp/A3_Q3.yaml --name q35_v0     # SFT rows (src/train/sft_data.py)
    python -m cli kg-build                                    # legal KG -> data/kg/kg.sqlite
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
    v = sub.add_parser("vbpl", help="extra corpus from vbpl.vn (HF th1nhng0/vietnamese-legal-documents)")
    v.add_argument("--cited", action="store_true", help="M2: >= 2000 + every cited doc -> data/processed/vbpl_cited")
    eb = sub.add_parser("embed-bench", help="compare embedders on a small pool (dense top-50 + gold articles): Điều@k, MRR, latency (src/evaluate/embed_bench.py)")
    eb.add_argument("exp")
    eb.add_argument("--split", default="dev_clean")
    eb.add_argument("--models", default="configs/embed_bench.yaml")
    eb.add_argument("--out", default="data/runs/embed_bench.json")
    eb.add_argument("--limit", type=int, default=0)
    eb.add_argument("--export", default=None, help="only build the pool and save it as JSON (for a remote job)")
    sub.add_parser("kg-build", help="legal KG (rules, no LLM) from vbpl metadata + relationships -> data/kg/kg.sqlite (src/kg)")
    sub.add_parser("web", help="cited documents missing from course + vbpl, from luatvietnam.vn -> data/processed/web")
    x = sub.add_parser("extra", help="public legal datasets for training -> data/raw/extra (see corpus/extra.py)")
    x.add_argument("names", nargs="*")
    d = sub.add_parser("sft-data", help="decontaminated SFT rows (prompt as at inference + reference answer) -> data/train/<name>")
    d.add_argument("exp")
    d.add_argument("--name", required=True)
    d.add_argument("--splits", default="train", help="comma list; the final model uses train,dev")
    d.add_argument("--k-range", default="", help="'5,10': k drawn per row (needs a retrieval cache for k=max)")
    d.add_argument("--oracle", type=float, default=0.0, help="p of injecting a gold passage when top-k misses it (RAFT)")
    d.add_argument("--shots", default="0", help="comma list of shot counts drawn per row, e.g. 0,1,2,3")
    d.add_argument("--refuse", type=float, default=0.0, help="p of an extra no-answer row (unrelated context -> refusal)")
    d.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    es = sub.add_parser("export-shots", help="few-shot pool (decontaminated train) -> JSON, so the QA container needs no dataset")
    es.add_argument("--out", default="data/serve/shots.json")
    rr = sub.add_parser("rerank-data", help="reranker groups (gold Điều positives, dense hard negatives, Zalo) -> data/train/<name>")
    rr.add_argument("exp")
    rr.add_argument("--name", required=True)
    rr.add_argument("--no-zalo", action="store_true")
    rr.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r = sub.add_parser("run")
    r.add_argument("exp")
    r.add_argument("--backend")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--no-score", action="store_true")
    r.add_argument("--retrieval-only", action="store_true", help="retrieve (cached) + doc@k / Điều@k; no LLM, no run dir")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--no-cosine", action="store_true")
    g = sub.add_parser("retrieval", help="doc@k / Điều@k of a run's (or cache file's) passages vs corpus.gold")
    g.add_argument("target", help="data/runs/<run> or data/cache/retrieval/<key>.jsonl")
    g.add_argument("--rebuild-gold", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--metric", default="rouge_l")
    b = sub.add_parser("box", help="remote GPU box: push / setup / serve / run / status / stop / pull / tunnel")
    b.add_argument("action", choices=["push", "push-data", "setup", "serve", "run", "status", "stop", "pull", "tunnel",
                                      "watch", "autosync", "queue"])
    b.add_argument("target", nargs="?", help="serve: model id · run: exp yaml · queue: file, one exp.yaml + args per line · stop: session (llm-g0, embed-g0, run-<exp>) · autosync: on|off")
    b.add_argument("--gpu", type=int, default=0)
    b.add_argument("--kind", default="llm", choices=["llm", "embed", "rerank"], help="serve: which vLLM server")
    b.add_argument("--parallel", action="store_true", help="queue: start now, beside running queues (else wait for them)")
    b.add_argument("--on", default="l40s", help="backend file with a `box:` block")
    args, extra = ap.parse_known_args()
    if extra and args.cmd != "box":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.cmd == "data":
        from corpus import DatasetBuilder

        DatasetBuilder().build()
    elif args.cmd == "vbpl":
        from corpus.sources.citations import Citations
        from corpus.sources.vbpl import VbplImporter

        if args.cited:
            VbplImporter(2000, Citations().table()["vbpl_id"].drop_nulls().to_list(), "data/processed/vbpl_cited").build()
        else:
            VbplImporter().build()
    elif args.cmd == "embed-bench":
        from evaluate.embed_bench import run_models

        run_models(args.exp, args.split, args.models, args.out, args.limit, args.export)
    elif args.cmd == "kg-build":
        from kg import KG

        print(KG.build().counts())
    elif args.cmd == "web":
        from corpus.sources.citations import Citations
        from corpus.sources.luatvietnam import LuatVietnam

        LuatVietnam().build(Citations().table().filter(where="web").to_dicts())
    elif args.cmd == "extra":
        from corpus.sources.extra import ExtraData

        ExtraData().download(args.names)
    elif args.cmd == "sft-data":
        from config import load
        from train.sft_data import SFTData

        kr = tuple(int(x) for x in args.k_range.split(",")) if args.k_range else None
        SFTData(load(args.exp, args.set), args.name, tuple(args.splits.split(",")), kr, args.oracle,
                shots=tuple(int(x) for x in args.shots.split(",")), refuse=args.refuse).build()
    elif args.cmd == "export-shots":
        from generate.shots import ShotStore

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        store = ShotStore()
        store.save(args.out)
        print(f"{len(store.pool)} shots -> {args.out}")
    elif args.cmd == "rerank-data":
        from config import load
        from train.rerank_data import RerankData

        RerankData(load(args.exp, args.set), args.name, zalo=not args.no_zalo).build()
    elif args.cmd == "run":
        from config import load
        from pipeline import Experiment

        if args.retrieval_only:
            from corpus import QASet
            from evaluate import RetrievalEval
            from retrieve import Retrieval

            cfg = load(args.exp, args.set, args.backend)
            r = Retrieval(cfg)
            qids = set(r.run(QASet().load(cfg.split, cfg.limit)))
            print(json.dumps({"split": cfg.split, "k": cfg.retrieve.k, **RetrievalEval().score(r.path, qids)}))
            return
        summary = Experiment(load(args.exp, args.set, args.backend)).execute(score=not args.no_score)
        print(json.dumps(summary, indent=1) if summary else "answers written; score later with `cli score`")
    elif args.cmd == "score":
        from evaluate import Evaluator
        from runs import Run

        print(json.dumps(Evaluator(cosine=not args.no_cosine).score(Run(Path(args.run))), indent=1))
    elif args.cmd == "retrieval":
        from corpus.gold import GoldSet
        from evaluate import RetrievalEval

        if args.rebuild_gold:
            GoldSet().build()
        t = Path(args.target)
        cache, qids = (t, None) if t.suffix == ".jsonl" else RetrievalEval.cache_of(t)
        print(json.dumps(RetrievalEval().score(cache, qids), indent=1))
    elif args.cmd == "compare":
        from evaluate import PairedBootstrap
        from runs import Run

        a, b = (({r["qid"]: r[args.metric] for r in Run(Path(p)).read_json("metrics.json")["rows"]}) for p in (args.a, args.b))
        print(json.dumps(PairedBootstrap().compare(a, b), indent=1))

    elif args.cmd == "box":
        from ops import RemoteBox

        box = RemoteBox.from_backend(args.on)
        needs_target = {"serve", "run", "stop", "autosync", "queue"}
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
            "queue": lambda: box.queue([ln.split() for ln in Path(args.target).read_text().splitlines()
                                        if ln.strip() and not ln.startswith("#")], args.gpu, args.parallel),
        }[args.action]()


if __name__ == "__main__":
    main()
