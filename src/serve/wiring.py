"""Glue between the container environment and the service: env -> config overrides, endpoint probes, building the QAService."""
from pathlib import Path

ENV_TO_KEY = {"LLM_URL": "backend.base_url", "EMBED_URL": "backend.embed_url", "RERANK_URL": "backend.rerank_url"}


def overrides_from_env(env) -> list[str]:
    """LLM_URL / EMBED_URL / RERANK_URL (any OpenAI-compatible or vLLM pooling endpoint, local or shared) -> `--set` overrides."""
    return [f"{key}={env[name]}" for name, key in ENV_TO_KEY.items() if env.get(name)]


def probe(urls: dict[str, str], get=None, timeout: float = 3.0) -> dict[str, bool]:
    """One health check per inference endpoint: GET <server root>/health (vLLM serves it at the root, not under /v1)."""
    if get is None:
        import httpx

        get = httpx.get
    out = {}
    for name, url in urls.items():
        root = url.rstrip("/").removesuffix("/v1")
        try:
            out[name] = get(f"{root}/health", timeout=timeout).status_code == 200
        except Exception:  # noqa: BLE001 — refused / timed out / DNS: the endpoint is down
            out[name] = False
    return out


def build_service(cfg, shots_path: str | Path, min_score: float = 0.0):
    """Retriever + reranker (calling the embed / rerank endpoints), Generator (LLM endpoint) with the shot pool loaded from JSON."""
    from generate import Generator
    from generate.shots import ShotStore
    from index import IndexStore
    from retrieve.factory import RetrieverFactory
    from serve.qa_service import QAService

    factory = RetrieverFactory(cfg, IndexStore(cfg))
    generator = Generator(cfg.generate, cfg.backend)
    if cfg.generate.shots:
        generator._shots = ShotStore.load(shots_path)
    return QAService(factory.retriever(), factory.reranker(), generator, k=cfg.retrieve.k, min_score=min_score)


def limiter_from_env(env):
    """MAX_CONCURRENT / MAX_QUEUE; unset or empty (docker compose passes "") = unlimited."""
    from serve.limiter import Limiter

    return Limiter(int(env.get("MAX_CONCURRENT") or 0) or None, int(env["MAX_QUEUE"]) if env.get("MAX_QUEUE") else None)
