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


class StreamService(FakeService):
    async def stream(self, question: str):
        yield {"event": "sources", "sources": [{"doc": "A", "text": "Điều 1", "score": 0.5}]}
        for t in ["Xin ", "chào"]:
            yield {"event": "token", "text": t}
        yield {"event": "done", "answer": "Xin chào", "refused": False, "timings": {"ttft_s": 0.2, "total_s": 0.5}}


def sclient(**kw):
    return TestClient(create_app(StreamService(), health=lambda: {"llm": True}, **kw))


def sse(text: str) -> list[tuple[str, dict]]:
    import json

    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(ln.split(": ", 1) for ln in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_ask_stream_is_server_sent_events():
    r = sclient().post("/ask/stream", json={"question": "q"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse(r.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "done"]
    assert events[-1][1]["answer"] == "Xin chào"


def test_openai_chat_completion_uses_the_last_user_message():
    r = sclient().post("/v1/chat/completions", json={"model": "law-rag", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "Mức phạt?"}]})
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Xin chào"}
    assert body["sources"][0]["doc"] == "A"


def test_openai_chat_completion_streaming_chunks_and_done_marker():
    import json

    r = sclient().post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "q"}]})
    lines = [ln[len("data: "):] for ln in r.text.strip().split("\n\n")]
    assert lines[-1] == "[DONE]"
    deltas = [json.loads(x)["choices"][0]["delta"].get("content") for x in lines[:-1]]
    assert "".join(d for d in deltas if d) == "Xin chào"
    assert json.loads(lines[-2])["choices"][0]["finish_reason"] == "stop"


def test_openai_endpoint_rejects_a_request_without_user_message():
    assert sclient().post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "s"}]}).status_code == 422


def test_metrics_endpoint_counts_requests():
    c = sclient()
    c.post("/ask", json={"question": "q"})
    text = c.get("/metrics").text
    assert 'lawrag_requests_total{route="/ask",status="200"} 1' in text
    assert "lawrag_latency_seconds_count" in text


def test_overloaded_service_answers_429():
    from serve.limiter import Limiter, Overloaded

    class Full(Limiter):
        async def __aenter__(self):
            raise Overloaded("full")

    r = sclient(limiter=Full()).post("/ask", json={"question": "q"})
    assert r.status_code == 429 and r.headers["retry-after"]
