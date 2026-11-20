from fastapi.testclient import TestClient

from serve.api import create_app


class FakeService:
    async def ask(self, question: str) -> dict:
        if question == "boom":
            raise RuntimeError("llm endpoint down")
        return {"answer": f"trả lời: {question}", "sources": [{"doc": "A", "text": "Điều 1", "score": 0.5}],
                "timings": {"retrieve_s": 0.1, "rerank_s": 0.05, "generate_s": 1.0, "total_s": 1.15}}


def client(health=None):
    return TestClient(create_app(FakeService(), health=health or (lambda: {"llm": True, "embed": True, "rerank": True})))


def test_ask_returns_answer_sources_and_timings():
    r = client().post("/ask", json={"question": "Mức phạt là bao nhiêu?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "trả lời: Mức phạt là bao nhiêu?" and body["sources"][0]["doc"] == "A" and body["timings"]["total_s"] == 1.15


def test_empty_question_is_rejected():
    assert client().post("/ask", json={"question": "   "}).status_code == 422
    assert client().post("/ask", json={}).status_code == 422


def test_a_failing_endpoint_becomes_503_not_a_crash():
    r = client().post("/ask", json={"question": "boom"})
    assert r.status_code == 503 and "llm endpoint down" in r.json()["detail"]


def test_health_reports_each_endpoint_and_503_when_one_is_down():
    assert client().get("/health").json() == {"status": "ok", "endpoints": {"llm": True, "embed": True, "rerank": True}}
    down = client(health=lambda: {"llm": True, "embed": False, "rerank": True}).get("/health")
    assert down.status_code == 503 and down.json()["endpoints"]["embed"] is False
