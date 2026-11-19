"""Modal job: compare embedders (src/evaluate/embed_bench.py) on a pool exported by `cli embed-bench --export`.

Account/profile is chosen outside: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=<p> modal run --detach src/ops/modal_embed.py::bench`
Volumes (per account, created on first use): law-rag-embed-data (pool.json + configs/embed_bench.yaml in), law-rag-embed-results (results out),
law-rag-embed-hf (model cache; big models are purged after use). One container, one L40S (48 GB), timeout = hard cost cap (~2 USD/h).

    modal volume put law-rag-embed-data data/runs/embed_pool_dev_clean.json /pool.json
    modal volume put law-rag-embed-data configs/embed_bench.yaml /embed_bench.yaml
"""
import importlib.util
import json
import sys
from pathlib import Path

import modal

app = modal.App("law-rag-embed")
data_vol = modal.Volume.from_name("law-rag-embed-data", create_if_missing=True)
res_vol = modal.Volume.from_name("law-rag-embed-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("law-rag-embed-hf", create_if_missing=True)
_here = Path(__file__).resolve()
SRC = _here.parents[1] if _here.parent.name == "ops" else Path("/src")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers>=4.51,<5", "sentence-transformers>=3.4,<6", "numpy<2", "pyyaml", "accelerate", "einops")
         .add_local_dir(SRC, "/src"))


@app.function(image=image, gpu="L40S", timeout=3600, volumes={"/data": data_vol, "/results": res_vol, "/hf": hf_vol},
              env={"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
def bench(pool: str = "/data/pool.json", specs: str = "/data/embed_bench.yaml", out: str = "/results/embed_bench.json") -> str:
    sys.path.insert(0, "/src")
    spec = importlib.util.spec_from_file_location("embed_bench", "/src/evaluate/embed_bench.py")   # by path: evaluate/__init__ would import LlamaIndex
    eb = importlib.util.module_from_spec(spec)
    sys.modules["embed_bench"] = eb
    spec.loader.exec_module(eb)
    passages, queries, gold, keys, groups = eb.load_pool(pool)
    bench = eb.EmbedBench(passages, queries, gold, keys=keys, groups=groups)
    print(f"pool {len(passages)} passages, {len(queries)} questions", flush=True)
    rows = eb.run_specs(bench, eb.load_specs(specs), lambda s: eb.STEmbedder(s["model"], s["max_len"]), purge_home="/hf")
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    res_vol.commit()
    return json.dumps(rows, ensure_ascii=False)


@app.local_entrypoint()
def main():
    print(bench.remote())
