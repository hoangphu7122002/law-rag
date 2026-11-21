"""REST layer. Knows nothing about models, only the QAService it is given.

  POST /ask                  {"question": ...} -> answer, sources, timings
  POST /ask/stream           same, as server-sent events: sources, token*, done
  POST /v1/chat/completions  OpenAI-compatible (stream or not): the last user message is the question; `sources` is an extra field
  GET  /health, GET /metrics (Prometheus text)

Admission control: a Limiter (max concurrent, bounded queue) answers 429 + Retry-After instead of letting a burst slow everyone.
"""
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from serve.limiter import Limiter, Overloaded
from serve.metrics import Metrics


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "law-rag"
    stream: bool = False


def last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"]).strip()
    raise HTTPException(status_code=422, detail="no user message")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service, health, limiter: Limiter | None = None, metrics: Metrics | None = None) -> FastAPI:
    """health: callable -> {"llm": bool, "embed": bool, "rerank": bool} (one probe per inference endpoint)."""
    app = FastAPI(title="law-rag QA", version="1.1")
    limiter, metrics = limiter or Limiter(), metrics or Metrics()

    def track(route: str, status: int, started: float, timings: dict | None = None, refused: bool = False) -> None:
        metrics.inc("requests_total", route=route, status=str(status))
        metrics.observe("latency_seconds", time.perf_counter() - started, route=route)
        if timings and "ttft_s" in timings:
            metrics.observe("ttft_seconds", timings["ttft_s"], route=route)
        if refused:
            metrics.inc("refusals_total")
        metrics.gauge("in_flight", limiter.running)

    async def admit(route: str):
        try:
            await limiter.__aenter__()
        except Overloaded as e:
            metrics.inc("requests_total", route=route, status="429")
            raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": "2"}) from e
        metrics.gauge("in_flight", limiter.running)

    async def release() -> None:
        await limiter.__aexit__(None, None, None)

    async def complete(question: str) -> dict:
        """Run the question to the end; uses the streaming path when the service has one (so /ask and /v1 agree)."""
        if not hasattr(service, "stream"):
            return await service.ask(question)
        sources, done = [], {}
        async for e in service.stream(question):
            if e["event"] == "sources":
                sources = e["sources"]
            elif e["event"] == "done":
                done = e
        return {"answer": done["answer"], "refused": done["refused"], "sources": sources, "timings": done["timings"]}

    @app.post("/ask")
    async def ask(req: AskRequest) -> dict:
        t0 = time.perf_counter()
        await admit("/ask")
        try:
            out = await service.ask(req.question)
        except Exception as e:  # noqa: BLE001 — an endpoint outage is a 503 for the caller, never a crashed worker
            track("/ask", 503, t0)
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}") from e
        finally:
            await release()
        track("/ask", 200, t0, out.get("timings"), out.get("refused", False))
        return out

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        t0 = time.perf_counter()
        await admit("/ask/stream")

        async def gen():
            done = {}
            try:
                async for e in service.stream(req.question):
                    if e["event"] == "done":
                        done = e
                    yield sse(e["event"], {k: v for k, v in e.items() if k != "event"})
                track("/ask/stream", 200, t0, done.get("timings"), done.get("refused", False))
            except Exception as e:  # noqa: BLE001
                track("/ask/stream", 503, t0)
                yield sse("error", {"detail": f"{type(e).__name__}: {e}"})
            finally:
                await release()

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        question = last_user_message(req.messages)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta: dict, finish=None, **extra) -> str:
            body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

        t0 = time.perf_counter()
        await admit("/v1/chat/completions")
        if not req.stream:
            try:
                out = await complete(question)
            except Exception as e:  # noqa: BLE001
                track("/v1/chat/completions", 503, t0)
                raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}") from e
            finally:
                await release()
            track("/v1/chat/completions", 200, t0, out["timings"], out["refused"])
            return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": out["answer"]}, "finish_reason": "stop"}],
                    "sources": out["sources"], "timings": out["timings"]}

        async def gen():
            sources, done = [], {}
            try:
                yield chunk({"role": "assistant"})
                async for e in service.stream(question):
                    if e["event"] == "sources":
                        sources = e["sources"]
                    elif e["event"] == "token":
                        yield chunk({"content": e["text"]})
                    elif e["event"] == "done":
                        done = e
                yield chunk({}, finish="stop", sources=sources, timings=done.get("timings", {}))
                yield "data: [DONE]\n\n"
                track("/v1/chat/completions", 200, t0, done.get("timings"), done.get("refused", False))
            except Exception as e:  # noqa: BLE001
                track("/v1/chat/completions", 503, t0)
                yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"
            finally:
                await release()

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "law-rag", "object": "model", "owned_by": "law-rag"}]}

    @app.get("/metrics")
    def prometheus():
        metrics.gauge("in_flight", limiter.running)
        metrics.gauge("queued", limiter.waiting)
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/health")
    def check():
        endpoints = health()
        ok = all(endpoints.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "endpoints": endpoints}, status_code=200 if ok else 503)

    return app
