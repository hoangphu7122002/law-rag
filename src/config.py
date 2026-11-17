"""Load one experiment: configs/base.yaml <- configs/exp/<x>.yaml <- --set overrides, plus the backend file.

    cfg = load("configs/exp/B0_baseline.yaml", overrides=["retrieve.k=10"])

Unknown keys are errors (typos fail loudly). Hashes name run directories and caches.
"""
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

CONFIGS = Path("configs")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Chunk(Strict):
    kind: Literal["recursive", "dieu"] = "recursive"
    size: int = 2000
    overlap: int = 50
    lowercase: bool = False
    flatten: bool = False


class Embed(Strict):
    model: str
    engine: str = "local"   # local (sentence-transformers in this process) | vllm (server at backend.embed_url)
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    max_len: int = 512
    batch: int = 128


class Retrieve(Strict):
    kind: Literal["dense", "bm25", "hybrid"] = "dense"
    metric: Literal["ip", "l2"] = "ip"
    k: int = 5
    candidates: int = 50
    rrf_k: int = 60


class Rerank(Strict):
    model: str | None = None
    engine: str = "local"   # local (CrossEncoder) | vllm (server at backend.rerank_url)
    max_len: int = 1024
    batch: int = 32


class Generate(Strict):
    model: str
    params_b: float
    mode: Literal["chat", "completion"] = "chat"
    prompt: str = "cancu"
    max_tokens: int = 768
    temperature: float = 0.0
    concurrency: int = 16

    @field_validator("params_b")
    @classmethod
    def under_4b(cls, v: float) -> float:
        if v >= 4.0:
            raise ValueError(f"answering LLM must have < 4B parameters, got {v}B")
        return v


class Box(Strict):
    """A remote GPU machine driven over ssh (src/ops/box.py)."""
    ssh: str
    root: str
    vllm: str
    hf_home: str
    min_free_mib: int = 20000          # LLM server; pooling servers need gpu_share[kind] of the card instead
    gpu_share: dict[str, float] = {"llm": 0.45, "embed": 0.15, "rerank": 0.15}
    llm_args: list[str] = []           # extra `vllm serve` flags for the LLM server
    pooling_args: list[str] = []       # extra flags for embed / rerank servers
    env: dict[str, str] = {}  # exported for every job on the box (serve and run); {root} expands


class Backend(Strict):
    name: str
    base_url: str
    api_key: str = "EMPTY"
    device: str = "cpu"
    embed_url: str | None = None    # vLLM pooling server for embed.engine=vllm
    rerank_url: str | None = None   # vLLM pooling server for rerank.engine=vllm
    box: Box | None = None


def load_backend(name: str) -> Backend:
    return Backend(**_read(CONFIGS / "backends" / f"{name}.yaml"))


class Config(Strict):
    name: str
    note: str = ""
    split: Literal["train", "dev", "test"] = "dev"
    limit: int | None = None
    corpus_limit: int | None = None
    sources: list[str] = ["corpus"]
    seed: int = 20261031
    backend: Backend
    chunk: Chunk
    embed: Embed
    retrieve: Retrieve
    rerank: Rerank
    generate: Generate

    # --- hashes: same inputs -> same name, so reruns resume and caches are reused ---
    def index_key(self) -> str:
        extra = [] if self.sources == ["corpus"] else [self.sources]  # keeps the hash of course-corpus indexes
        if self.embed.engine != "local":
            extra.append(self.embed.engine)
        return _hash(self.corpus_limit, *extra, self.chunk, self.embed.model, self.embed.passage_prefix, self.embed.normalize, self.embed.max_len,
                     self.retrieve.metric)

    def retrieval_key(self) -> str:
        return _hash(self.index_key(), self.embed, self.retrieve, self.rerank, self.split, self.limit)

    def run_key(self) -> str:
        return _hash(self.retrieval_key(), self.generate)


def load(exp: str | Path, overrides: list[str] = (), backend: str | None = None) -> Config:
    exp = Path(exp)
    data = _merge(_read(CONFIGS / "base.yaml"), _read(exp))
    data["backend"] = _read(CONFIGS / "backends" / f"{backend or data['backend']}.yaml")  # same file as load_backend()
    for item in overrides:  # after the backend file, so --set backend.base_url=... works too
        key, _, value = item.partition("=")
        _set(data, key, yaml.safe_load(value))
    return Config(name=exp.stem, **data)


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set(data: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    for p in parents:
        data = data.setdefault(p, {})
    data[last] = value


def _hash(*parts) -> str:
    def plain(x):
        return x.model_dump() if isinstance(x, BaseModel) else x
    blob = json.dumps([plain(p) for p in parts], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]
