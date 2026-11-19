"""Load one experiment: configs/base.yaml <- configs/exp/<x>.yaml <- --set overrides, plus the backend file.

    cfg = load("configs/exp/B0_baseline.yaml", overrides=["retrieve.k=10"])

Unknown keys are errors (typos fail loudly). Hashes name run directories and caches.
"""
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CONFIGS = Path("configs")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Chunk(Strict):
    kind: Literal["recursive", "dieu"] = "recursive"
    size: int = 2000
    overlap: int = 50
    lowercase: bool = False
    flatten: bool = False
    clean: bool = False   # LawCleaner before parsing (strip header / signature / tail blocks, title into metadata)
    dedup: bool = False   # DocDedup after cleaning: exact + same-number near-duplicate documents


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
    oracle: bool = False   # eval ceiling: inject a gold passage at rank 1 (retrieve/oracle.py); not part of retrieval_key


class Rerank(Strict):
    model: str | None = None
    engine: str = "local"   # local (CrossEncoder) | vllm (server at backend.rerank_url)
    max_len: int = 1024
    batch: int = 32


class WebCfg(Strict):
    """Web search tool settings (generate/web.py); the tool is on only when "web_search" is in generate.agent_tools."""
    provider: Literal["tinyfish"] = "tinyfish"
    domains: list[str] = []                    # allow-list of hosts (subdomains included); empty = no filter
    max_results: int = 5
    calls_per_minute: int = 30                 # TinyFish HOBBY (Search free, 30/min); Builder = 150/min but $0.005 per query (2026-11-19 billing page)
    calls_per_hour: int = 500
    cost_per_query: float = 0.0                # USD per paid search: Hobby 0, Builder 0.005
    max_cost_usd: float | None = None          # hard spending cap over all runs sharing this cache (None = no cap)
    key_file: str | None = None                # file holding the API key (env TINYFISH_API_KEY wins); keeps the key off command lines
    cache: str = "data/cache/web.sqlite"       # one external call per distinct query; reruns replay the cache
    block_test_answers: bool = True            # drop results sharing a 13-gram with a test answer (evaluation integrity)


class Generate(Strict):
    model: str
    params_b: float
    mode: Literal["chat", "completion"] = "chat"
    prompt: str = "cancu"
    max_tokens: int = 768
    temperature: float = 0.0
    concurrency: int = 16
    # sampling (docs/notes/09-inference-configs): None = server / model default, not sent
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    thinking: bool | None = None   # Qwen3 / 3.5 switch; None for models without one (Vi-Qwen2)
    think_tokens: int = 1024       # cap on the thought when thinking is on, then "</think>" is forced
    check: list[str] = []          # optional answer gate (generate/check.py): failing candidates are dropped before the MBR pick; [] = off
    mbr: list[float] = []          # sample temperatures; non-empty -> MBR pick by ROUGE-L among the samples

    shots: int = 0                 # few-shot: nearest decontaminated train Q/A pairs prepended to the user turn (generate/shots.py)

    agent: bool = False            # ReAct loop with tools (generate/react.py) instead of one call
    max_turns: int = 4             # agent: model calls per question (the last one has no tools)
    agent_tools: list[str] = ["search_law", "get_article"]   # agent: which tools are offered (ablation arms; "kg_related" needs data/kg/kg.sqlite)
    tool_prompt: Literal["full", "compact"] = "full"   # agent: "compact" drops the per-tool prose list (the chat template already renders the JSON schemas)
    agent_seed: int = 0            # agent: 0 = the original per-question sampling seed; other values give independent trajectories for the same question
    web: WebCfg = WebCfg()
    agent_initial_k: int = -1      # agent: passages from the cached retrieval put in the first prompt (-1 = all k, 0 = none)

    teacher: bool = False          # >= 4B allowed only to measure a distillation teacher (KD0); never a reported result

    @model_validator(mode="after")
    def under_4b(self) -> "Generate":
        if self.params_b >= 4.0 and not self.teacher:
            raise ValueError(f"answering LLM must have < 4B parameters, got {self.params_b}B (teacher: true for KD0 only)")
        return self


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
    split: Literal["train", "dev", "dev_clean", "dev_strict", "test"] = "dev"
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
        chunk = self.chunk.model_dump(exclude={k for k in ("clean", "dedup") if not getattr(self.chunk, k)})  # old keys unchanged
        return _hash(self.corpus_limit, *extra, chunk, self.embed.model, self.embed.passage_prefix, self.embed.normalize, self.embed.max_len,
                     self.retrieve.metric)

    def retrieval_key(self) -> str:
        return _hash(self.index_key(), self.embed, self.retrieve.model_dump(exclude={"oracle"}), self.rerank, self.split, self.limit)

    def run_key(self) -> str:
        new = {"top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "thinking", "think_tokens", "mbr", "check", "teacher", "shots", "agent", "max_turns", "agent_initial_k", "agent_tools", "tool_prompt", "agent_seed", "web"}
        g = self.generate.model_dump(exclude={k for k in new if getattr(self.generate, k) == Generate.model_fields[k].default})
        return _hash(self.retrieval_key(), g, *(["oracle"] if self.retrieve.oracle else []))  # unset new fields left out: old run keys unchanged


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
