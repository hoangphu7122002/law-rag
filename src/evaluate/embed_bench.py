"""Compare embedders on a SMALL corpus (candidates + gold articles of the evaluation questions) instead of re-indexing 724k chunks.

    bench = EmbedBench(passages, queries={qid: text}, gold={qid: {"<law key>#<Điều>"}})
    bench.run(embedder)  ->  {"art@1": .., "art@3": .., "art@5": .., "art@20": .., "mrr": .., "n": ..}

`embedder(list[str]) -> ndarray (n, d)`; cosine similarity over the pool; ranks are aggregated to the article level (law key + Điều).
Caveat (docs/research/model-selection): a pool built from one retriever's candidates favours that retriever; winners must be re-checked on the full corpus.
"""
import time

import numpy as np

from records import Passage

KS = (1, 3, 5, 20)


def article_keys(passages: list[Passage]) -> list[str | None]:
    """"<law key>#<Điều>" of each passage (None without a heading). Needs the corpus helpers (LlamaIndex); a remote job gets these keys precomputed."""
    from evaluate.retrieval_eval import RetrievalEval

    out = []
    for p in passages:
        law, art = RetrievalEval.passage_keys({"doc": p.doc, "text": p.text})
        out.append(f"{law}#{art}" if art else None)
    return out


class EmbedBench:
    def __init__(self, passages: list[Passage], queries: dict[int, str], gold: dict[int, set[str]], clock=time.perf_counter,
                 keys: list[str | None] | None = None, groups: dict[str, list[int]] | None = None):
        self.passages, self.queries, self.gold, self.clock, self.groups = passages, queries, gold, clock, groups or {}
        self.keys = keys if keys is not None else article_keys(passages)

    def run(self, embedder, query_prefix: str = "", doc_prefix: str = "", batch: int = 64, query_embedder=None) -> dict[str, float]:
        """Prefixes: e5 wants "query: " / "passage: ", Qwen3-Embedding an instruction on the query side only.
        `query_embedder` = a different tower for the questions (query-only tuning keeps the document vectors, so the index stays valid)."""
        query_embedder = query_embedder or embedder
        t0 = self.clock()
        docs = _encode(embedder, [doc_prefix + p.text for p in self.passages], batch)
        passages_per_s = len(self.passages) / max(self.clock() - t0, 1e-9)
        qids = [q for q in self.queries if q in self.gold]
        qs = _encode(query_embedder, [query_prefix + self.queries[q] for q in qids], batch)
        single = []                                          # latency of ONE query alone (what a user waits for)
        for q in qids[:20]:
            t = self.clock()
            query_embedder([query_prefix + self.queries[q]])
            single.append((self.clock() - t) * 1000)
        docs, qs = _unit(docs), _unit(qs)
        pos_of = {}
        for i, q in enumerate(qids):
            order = np.argsort(-(docs @ qs[i]), kind="stable")
            ranked = []                                   # articles in rank order, each once
            for j in order:
                k = self.keys[j]
                if k and k not in ranked:
                    ranked.append(k)
            pos_of[q] = next((r for r, k in enumerate(ranked, 1) if k in self.gold[q]), None)
        out = {**_agg(pos_of, qids), "passages_per_s": passages_per_s, "query_ms_p50": float(np.median(single)) if single else 0.0}
        out["ranks"] = pos_of                               # per question (None = gold Điều not retrieved): paired tests need these
        out["groups"] = {g: _agg(pos_of, [q for q in members if q in pos_of]) for g, members in self.groups.items()}
        return out


def paired_bootstrap(ranks_a: dict, ranks_b: dict, metric: str = "art@1", n: int = 2000, seed: int = 20261031) -> dict:
    """Paired bootstrap of metric(A) - metric(B) over the questions both runs have. metric: "art@K" (gold Điều in top K) or "mrr".
    Returns diff, 95 % interval, one-sided p (share of resamples with diff <= 0) and n."""
    qs = sorted(set(ranks_a) & set(ranks_b), key=str)
    if metric == "mrr":
        val = lambda r: 1.0 / r if r else 0.0
    else:
        k = int(metric.split("@")[1])
        val = lambda r: float(r is not None and r <= k)
    d = np.array([val(ranks_a[q]) - val(ranks_b[q]) for q in qs])
    if not len(d):
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0, "n": 0}
    boots = np.random.default_rng(seed).choice(d, size=(n, len(d))).mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "p": float((boots <= 0).mean()), "n": len(d)}


def _agg(pos_of: dict[int, int | None], qids: list[int]) -> dict[str, float]:
    pos = [pos_of[q] for q in qids]
    hit = lambda k: float(np.mean([p is not None and p <= k for p in pos])) if pos else 0.0
    return {**{f"art@{k}": hit(k) for k in KS}, "mrr": float(np.mean([1.0 / p if p else 0.0 for p in pos])) if pos else 0.0, "n": len(pos)}


def _encode(embedder, texts: list[str], batch: int) -> np.ndarray:
    return np.asarray([v for i in range(0, len(texts), batch) for v in embedder(texts[i:i + batch])], dtype=float)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def build_pool(candidates: dict[int, list[Passage]], gold: dict[int, set[str]], articles: dict[str, list[Passage]]) -> list[Passage]:
    """Unique passages: every question's retrieved candidates (in order), then the gold articles the candidates did not contain.
    `articles` maps "<law key>#<Điều>" to the corpus passages of that article (a gold article absent from the corpus is skipped)."""
    seen, pool = set(), []
    for ps in candidates.values():
        for p in ps:
            if p.pid not in seen:
                seen.add(p.pid)
                pool.append(p)
    for pairs in gold.values():
        for g in sorted(pairs):
            for p in articles.get(g, []):
                if p.pid not in seen:
                    seen.add(p.pid)
                    pool.append(p)
    return pool


class ServerEmbedder:
    """Embeddings from an OpenAI-compatible server (vLLM `--runner pooling`): big models run in their own server, not in this process."""

    def __init__(self, client, model: str, dims: int | None = None, max_len: int | None = None):
        self.client, self.model, self.dims, self.max_len = client, model, dims, max_len

    def __call__(self, texts: list[str]) -> np.ndarray:
        kw = {"extra_body": {"truncate_prompt_tokens": self.max_len}} if self.max_len else {}
        r = self.client.embeddings.create(model=self.model, input=list(texts), **kw)
        v = np.asarray([d.embedding for d in r.data], dtype=float)
        return v[:, : self.dims] if self.dims else v   # Matryoshka-style truncation (models trained for it)


DEFAULTS = {"purge": False, "kind": "st", "query_prefix": "", "doc_prefix": "", "batch": 32, "max_len": None, "dims": None, "base_url": None}


def load_specs(path) -> list[dict]:
    import yaml

    return [{**DEFAULTS, **s} for s in yaml.safe_load(open(path, encoding="utf-8"))]


class STEmbedder:
    """sentence-transformers in this process (fp16 on the GPU, loaded on first call); for models up to ~8B on one 46 GB card."""

    def __init__(self, model: str, max_len: int | None = None, trust_remote_code: bool = True):
        self.name, self.max_len, self.trust, self._m = model, max_len, trust_remote_code, None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self._m is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(self.name, device="cuda", trust_remote_code=self.trust, model_kwargs={"torch_dtype": torch.float16})
            if self.max_len:
                self._m.max_seq_length = self.max_len
        return self._m.encode(texts, batch_size=len(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def close(self) -> None:
        import gc

        import torch

        self._m = None
        gc.collect()
        torch.cuda.empty_cache()


def run_specs(bench: "EmbedBench", specs: list[dict], make, purge_home: str | None = None) -> list[dict]:
    """One row per spec; a model that fails (remote code, OOM, missing prefix support) becomes {"error": ...} and the run goes on.
    `make(spec)` builds the embedder; it is closed afterwards and, with `purge_home`, its cache is deleted when the spec says purge."""
    rows = []
    for s in specs:
        emb = None
        try:
            emb = make(s)
            m = bench.run(emb, s.get("query_prefix", ""), s.get("doc_prefix", ""), s.get("batch", 32))
            rows.append({"name": s["name"], "model": s["model"], **m})
        except Exception as e:  # noqa: BLE001
            rows.append({"name": s["name"], "model": s["model"], "error": repr(e)[:300]})
        finally:
            if emb is not None and hasattr(emb, "close"):
                emb.close()
            if purge_home and s.get("purge"):
                purge_model(s["model"], purge_home)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
    return rows


def run_models(exp: str, split: str, specs_path: str, out: str, limit: int = 0, export: str | None = None) -> None:
    """Orchestration for `cli embed-bench` (needs data + a GPU; not unit-tested): pool = dense top-50 of `exp` + gold articles, one model at a time."""
    import json
    import os
    from pathlib import Path

    from config import load
    from corpus.qa import QASet
    from retrieve import Retrieval

    cfg = load(exp)
    cfg = cfg.model_copy(update={"split": split, "rerank": cfg.rerank.model_copy(update={"model": None}),
                                 "retrieve": cfg.retrieve.model_copy(update={"k": cfg.retrieve.candidates})})
    qs = QASet().load(split, limit or None)
    from evaluate.retrieval_eval import RetrievalEval

    g = RetrievalEval().gold
    gold = {q.qid: g[q.qid].pairs for q in qs if q.qid in g and g[q.qid].pairs}
    cands = Retrieval(cfg).run(qs)
    articles: dict[str, list[Passage]] = {}   # gold articles missing from the candidates (~3 % of questions) are NOT added: loading all 724k nodes was OOM-killed (16 GB cap)
    pool = build_pool(cands, gold, articles)
    groups = {n: [q.qid for q in QASet().load(n) if q.qid in gold] for n in ("dev_clean", "dev_strict")} if split == "dev" else {}
    bench = EmbedBench(pool, {q.qid: q.text for q in qs}, gold, groups=groups)
    if export:
        save_pool(export, pool, {q.qid: q.text for q in qs}, gold, bench.keys, groups)
        print(f"pool exported to {export}")
        return
    print(f"pool {len(pool)} passages, {len(gold)} questions")
    def make(s):
        return STEmbedder(s["model"], s["max_len"]) if s["kind"] == "st" else ServerEmbedder(
            __import__("openai").OpenAI(base_url=s["base_url"], api_key="EMPTY"), s["model"], s["dims"], s["max_len"])

    rows = run_specs(bench, load_specs(specs_path), make, purge_home=os.environ.get("HF_HOME") if os.environ.get("HF_HOME", "").startswith(str(Path.cwd())) else None)
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def hf_cache_dir(model: str, home) -> "Path":
    from pathlib import Path

    return Path(home) / "hub" / ("models--" + model.replace("/", "--"))


def purge_model(model: str, home) -> bool:
    """Delete one model's HF cache folder under `home` (our own HF_HOME, never the box's shared cache); False if it was not there."""
    import shutil

    d = hf_cache_dir(model, home)
    if not d.is_dir():
        return False
    shutil.rmtree(d)
    return True


def save_pool(path, pool: list[Passage], queries: dict[int, str], gold: dict[int, set[str]], keys: list[str | None] | None = None,
              groups: dict[str, list[int]] | None = None) -> None:
    """One JSON file = everything a remote job needs (no corpus, no index): passages, questions, gold articles."""
    import json
    from pathlib import Path

    Path(path).write_text(json.dumps({"passages": [{"pid": p.pid, "doc": p.doc, "text": p.text} for p in pool], "queries": {str(k): v for k, v in queries.items()},
                                      "gold": {str(k): sorted(v) for k, v in gold.items()},
                                      "keys": keys if keys is not None else article_keys(pool), "groups": groups or {}}, ensure_ascii=False))


def load_pool(path) -> tuple[list[Passage], dict[int, str], dict[int, set[str]], list[str | None], dict[str, list[int]]]:
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return ([Passage(p["pid"], p["doc"], p["text"], 0.0) for p in d["passages"]], {int(k): v for k, v in d["queries"].items()},
            {int(k): set(v) for k, v in d["gold"].items()}, d["keys"], d.get("groups", {}))
