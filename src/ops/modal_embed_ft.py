"""Modal job: fine-tune AITeamVN embedder variants (src/train/retrieval/embed_ft.py) in parallel and score each on the dev pool (src/evaluate/embed_bench.py).

    modal volume put law-rag-embed-data data/runs/rr_v0_train.jsonl /rr_v0_train.jsonl      # groups (course train + Zalo, decontaminated)
    modal run --detach src/ops/modal_embed_ft.py --variants lora32,lora8,full [--limit 64]   # one L40S container per variant, ~$2/h each
Results: volume law-rag-embed-results:/embed_ft_<variant>.json (EmbedBench row + train minutes). Models are not kept (merged weights 2 GB each,
re-train the winner on the box). Account/profile chosen outside, see modal_embed.py.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("law-rag-embed-ft")
data_vol = modal.Volume.from_name("law-rag-embed-data", create_if_missing=True)
res_vol = modal.Volume.from_name("law-rag-embed-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("law-rag-embed-hf", create_if_missing=True)
_here = Path(__file__).resolve()
SRC = _here.parents[1] if _here.parent.name == "ops" else Path("/src")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers>=4.51,<5", "sentence-transformers>=3.4,<6", "peft>=0.13", "datasets<3.2", "pyarrow<17", "numpy<2",
                      "pyyaml", "accelerate", "einops")
         .add_local_dir(SRC, "/src"))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@app.function(image=image, gpu="L40S", timeout=3600, volumes={"/data": data_vol, "/results": res_vol, "/hf": hf_vol},
              env={"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
def train_eval(variant: str, limit: int = 0) -> list:
    sys.path.insert(0, "/src")
    eft = _load("embed_ft", "/src/train/embed_ft.py")
    eb = _load("embed_bench", "/src/evaluate/embed_bench.py")
    t0 = time.time()
    model_dir = eft.BASE if variant == "base" else eft.train(variant, "/data/rr_v0_train.jsonl", f"/tmp/{variant}", limit=limit)
    minutes = round((time.time() - t0) / 60, 1)
    passages, queries, gold, keys, groups = eb.load_pool("/data/pool.json")
    bench = eb.EmbedBench(passages, queries, gold, keys=keys, groups=groups)
    v = {} if variant == "base" else eft.variant_args(variant)
    rows = []

    def score(label: str, path: str, query_only: bool = False) -> None:
        emb = eb.STEmbedder(path, 1024)
        base_emb = eb.STEmbedder(eft.BASE, 1024) if query_only else None   # documents = the base tower (the existing index)
        m = bench.run(base_emb or emb, query_embedder=emb if query_only else None)
        for e in (emb, base_emb):
            e and e.close()
        rows.append({"name": label, **v, "train_minutes": minutes, "limit": limit, **m})

    score(variant, model_dir, query_only=v.get("mode") == "query")
    for a in v.get("alphas", []):
        score(f"{variant}@{a}", eft.interpolated_model(eft.BASE, model_dir, a, f"/tmp/{variant}_a{a}"))
    Path(f"/results/embed_ft_{variant}{'_smoke' if limit else ''}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    res_vol.commit()
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    return rows


@app.local_entrypoint()
def main(variants: str = "lora32", limit: int = 0):
    names = variants.split(",")
    for rows in train_eval.map(names, kwargs={"limit": limit}):
        for r in rows:
            print({k: r[k] for k in ("name", "art@1", "art@3", "art@5", "mrr", "train_minutes")})
