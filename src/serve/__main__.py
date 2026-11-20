"""Container entry point: python -m serve   (env: LAWRAG_EXP, SHOTS_PATH, PORT, MIN_RERANK_SCORE (refuse below this best reranker score, default 0.001; 0 = off), LLM_URL, EMBED_URL, RERANK_URL)."""
import os

from config import load
from serve.api import create_app
from serve.wiring import build_service, overrides_from_env, probe


def main() -> None:
    import uvicorn

    env = os.environ
    cfg = load(env.get("LAWRAG_EXP", "configs/exp/S1_serve.yaml"), overrides_from_env(env))
    service = build_service(cfg, env.get("SHOTS_PATH", "/app/shots.json"), float(env.get("MIN_RERANK_SCORE", "0.001")))
    urls = {"llm": cfg.backend.base_url, "embed": cfg.backend.embed_url, "rerank": cfg.backend.rerank_url}
    uvicorn.run(create_app(service, health=lambda: probe(urls)), host="0.0.0.0", port=int(env.get("PORT", "8080")))


if __name__ == "__main__":
    main()
