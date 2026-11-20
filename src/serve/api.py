"""REST layer: POST /ask {"question": ...} and GET /health. Knows nothing about models, only the QAService it is given."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


def create_app(service, health) -> FastAPI:
    """health: callable -> {"llm": bool, "embed": bool, "rerank": bool} (one probe per inference endpoint)."""
    app = FastAPI(title="law-rag QA", version="1.0")

    @app.post("/ask")
    async def ask(req: AskRequest) -> dict:
        try:
            return await service.ask(req.question)
        except Exception as e:  # noqa: BLE001 — an endpoint outage is a 503 for the caller, never a crashed worker
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}") from e

    @app.get("/health")
    def check():
        endpoints = health()
        ok = all(endpoints.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "endpoints": endpoints}, status_code=200 if ok else 503)

    return app
