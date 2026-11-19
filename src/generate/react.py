"""ReAct answer loop on a RAW (untrained) chat model with native tool calls (datamate's recipe, docs/notes/13-react-baseline.md).

The model gets the usual RAG prompt (initial passages from the cached retrieval, `agent_initial_k` of them; 0 = none) and two tools:

  search_law(query)            new retrieval + rerank over the whole corpus, top `k` passages
  get_article(law, dieu)       the full text of Điều `dieu` of the document `law` ("50/2024/TT-NHNN")

No tool call in a reply = the final answer (Qwen3.5 native <tool_call> format; vLLM needs --enable-auto-tool-choice
--tool-call-parser qwen3_coder, set in configs/backends/l40s.yaml llm_args).

Loop guards (datamate part 5: ca1 lost 42 % of its turns to errors, repeats and context overflow):
  - an identical repeated call is not run again (the model is told to use what it has / answer);
  - on the last turn tools are removed and the model is told to answer now;
  - every tool result is cut to `RESULT_CHARS`; one seed per question (stable across reruns);
  - a tool error is returned to the model as text, never raised.
"""
import asyncio
import json
import re
import zlib

from config import Backend, Generate
from corpus.keys import law_key, stem_key
from generate.check import DIEU, Gate
from generate.generator import Generator
from records import Passage

RESULT_CHARS, ARTICLE_CHARS = 1800, 4000
HEADING = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zđ]?)\s*[.:]")

TOOL_HELP = {
    "search_law": "- search_law(query): tìm các đoạn luật liên quan đến một câu truy vấn.",
    "get_article": "- get_article(law, dieu): lấy toàn văn một Điều của một văn bản (ví dụ law=\"50/2024/TT-NHNN\", dieu=5).",
    "kg_related": "- kg_related(law, hops): liệt kê văn bản sửa đổi, thay thế, bãi bỏ một văn bản; thường chọn hops=1 hoặc 2, "
                  "số văn bản tăng rất nhanh theo hops nên chỉ tăng khi thật sự cần.",
    "web_search": "- web_search(query): tìm trên web (chỉ các trang pháp luật được phép); dùng khi các công cụ trong kho văn bản không có thông tin.",
}


def system_tools(tools: list[str]) -> str:
    return ("Bạn có thể dùng công cụ để tìm thêm thông tin nếu ngữ cảnh chưa đủ:\n" + "\n".join(TOOL_HELP[t] for t in tools) +
            "\nChỉ gọi công cụ khi ngữ cảnh hiện có chưa đủ. Khi đã đủ thông tin, trả lời trực tiếp, không gọi công cụ nữa, "
            "theo mẫu \"Căn cứ Điều X ... quy định ...\".")


MAX_HOPS = 4   # absolute safety bound on what the model may ask for (the model is only ADVISED to use 1-2; hop sizes explode)
TOOLS = [
    {"type": "function", "function": {
        "name": "search_law", "description": "Tìm các đoạn luật liên quan đến truy vấn trong toàn bộ kho văn bản.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_article", "description": "Lấy toàn văn một Điều của một văn bản pháp luật.",
        "parameters": {"type": "object", "properties": {"law": {"type": "string", "description": "số hiệu văn bản, ví dụ 50/2024/TT-NHNN"},
                                                        "dieu": {"type": "integer"}}, "required": ["law", "dieu"]}}},
]

KG_TOOL = {"type": "function", "function": {
    "name": "kg_related",
    "description": ("Tìm các văn bản sửa đổi, thay thế, bãi bỏ hoặc làm hết hiệu lực một văn bản, theo số bước (hops). "
                    "Thường dùng hops=1 hoặc 2: mỗi bước thêm làm số văn bản tăng rất nhanh; kết quả luôn cho biết số văn bản thật ở mỗi bước, "
                    "nếu quá nhiều thì giảm hops. Chỉ trả tiêu đề, không trả nội dung; dùng get_article để đọc nội dung."),
    "parameters": {"type": "object", "properties": {"law": {"type": "string", "description": "số hiệu văn bản, ví dụ 75/2015/QH13"},
                                                    "hops": {"type": "integer", "description": "số bước, mặc định 1"}}, "required": ["law"]}}}
WEB_TOOL = {"type": "function", "function": {
    "name": "web_search", "description": "Tìm trên web các trang pháp luật được phép; trả tiêu đề, đường dẫn và một đoạn trích ngắn.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
ALL_TOOLS = {t["function"]["name"]: t for t in [*TOOLS, KG_TOOL, WEB_TOOL]}
WEB_SNIPPET_CHARS = 200


class LawTools:
    """Live retrieval and article lookup over the same index the experiment uses. Built lazily (the index is large)."""

    def __init__(self, cfg, provider=None, blocked_texts: list[str] | None = None):
        self.cfg, self._retriever, self._reranker, self._articles = cfg, None, None, None
        self._provider, self._blocked, self._web = provider, blocked_texts, None

    @property
    def web(self):
        """Web search with the config's allow-list / cache / leak filter (built on first use; needs TINYFISH_API_KEY for the real provider)."""
        if self._web is None:
            from generate.web import TinyFish, WebSearch

            w = self.cfg.generate.web if hasattr(self.cfg, "generate") else self.cfg.web
            blocked = self._blocked
            if blocked is None and w.block_test_answers:
                from corpus.qa import QASet

                blocked = [q.answer for q in QASet().load("test")]
            self._web = WebSearch(self._provider or TinyFish(key_file=w.key_file), cache=w.cache, domains=w.domains, max_results=w.max_results, blocked_texts=blocked,
                                  calls_per_minute=w.calls_per_minute, calls_per_hour=w.calls_per_hour,
                                  cost_per_query=w.cost_per_query, max_cost_usd=w.max_cost_usd)
        return self._web

    def _build(self) -> None:
        from index import IndexStore
        from retrieve.factory import RetrieverFactory

        self.store = IndexStore(self.cfg)
        f = RetrieverFactory(self.cfg, self.store)
        self._retriever, self._reranker = f.retriever(), f.reranker()

    def search_law(self, query: str) -> list[Passage]:
        from llama_index.core import QueryBundle
        from llama_index.core.schema import MetadataMode

        if self._retriever is None:
            self._build()
        hits = self._retriever.retrieve(query)
        if self._reranker is not None:
            hits = self._reranker.postprocess_nodes(hits, query_bundle=QueryBundle(query))
        return [Passage(pid=h.node.node_id, doc=h.node.metadata.get("doc", ""), text=h.node.get_content(metadata_mode=MetadataMode.LLM),
                        score=float(h.score or 0.0)) for h in hits[: self.cfg.retrieve.k]]

    def related(self, law: str, hops: int) -> list[dict] | None:
        from kg import KG

        if getattr(self, "_kg", None) is None:
            self._kg = KG("data/kg/kg.sqlite")
        return self._kg.related(law_key(str(law)), hops)

    def article(self, law: str, dieu: int) -> str:
        if self._articles is None:
            if self._retriever is None:
                self._build()
            idx: dict[tuple[str, str], list[str]] = {}
            for n in self.store.nodes():
                m = HEADING.search(n.get_content()[:400])
                if m:
                    idx.setdefault((stem_key(n.metadata.get("doc", "")), m.group(1)), []).append(n.get_content())
            self._articles = idx
        parts = self._articles.get((law_key(str(law)), str(dieu)))
        return "\n".join(parts) if parts else ""


class ReactGenerator(Generator):
    def __init__(self, cfg: Generate, backend: Backend, tools: LawTools | None = None):
        super().__init__(cfg, backend)
        self.tools, self._oai = tools, None

    @property
    def oai(self):
        if self._oai is None:
            from openai import AsyncOpenAI

            self._oai = AsyncOpenAI(base_url=self.backend.base_url, api_key=self.backend.api_key or "EMPTY", timeout=600, max_retries=2)
        return self._oai

    def tool_specs(self) -> list[dict]:
        return [ALL_TOOLS[n] for n in self.cfg.agent_tools]

    async def answer(self, question: str, passages: list[Passage]) -> str:
        init = passages if self.cfg.agent_initial_k < 0 else passages[: self.cfg.agent_initial_k]
        msgs = self.prompt.messages(question, init, self.shots(question))
        if msgs[0]["role"] == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + system_tools(self.cfg.agent_tools)}
        else:
            msgs.insert(0, {"role": "system", "content": system_tools(self.cfg.agent_tools)})
        seed = zlib.crc32(question.encode()) & 0x7FFFFFFF
        seen: set[str] = set()
        gate, repaired, context = Gate(self.cfg.check), False, list(init)   # context = what the model has seen (passages + tool results)
        for turn in range(self.cfg.max_turns):
            last = turn == self.cfg.max_turns - 1
            if last and turn > 0:
                msgs.append({"role": "user", "content": "Hãy trả lời câu hỏi ngay bây giờ dựa trên thông tin đã có, không gọi công cụ."})
            kw = self.sampling()
            extra = kw.pop("extra_body", {})
            r = await self.oai.chat.completions.create(
                model=self.cfg.model, messages=msgs, max_tokens=self.cfg.max_tokens, seed=seed, extra_body=extra,
                **({} if last else {"tools": self.tool_specs()}), **{k: v for k, v in kw.items()})
            m = r.choices[0].message
            calls = [] if last else (m.tool_calls or [])
            if not calls:
                text = self.clean(m.content or "")
                if gate.names and not repaired and not gate.passes(text, context) and turn < self.cfg.max_turns - 1:
                    repaired = True   # optional gate: one nudge, then the next answer is final
                    bad = sorted({n for n in DIEU.findall(text) if not Gate(["grounded_citation"]).passes(f"Điều {n}", context)})
                    msgs += [{"role": "assistant", "content": m.content or ""},
                             {"role": "user", "content": "Câu trả lời nhắc tới " + ", ".join(f"Điều {n}" for n in bad) + " nhưng điều đó không có trong ngữ cảnh hay kết quả "
                              "công cụ. Hãy kiểm tra bằng get_article hoặc trả lời lại chỉ dựa trên các điều luật đã có."}]
                    continue
                return text
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                key = c.function.name + c.function.arguments
                out = ("Bạn đã gọi đúng công cụ này với cùng tham số rồi; hãy dùng kết quả trước đó hoặc trả lời." if key in seen
                       else await asyncio.to_thread(self.run_tool, c.function.name, c.function.arguments))
                seen.add(key)
                context.append(Passage("tool", "", out, 0.0))
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
        return ""

    def run_tool(self, name: str, arguments: str) -> str:
        try:
            a = json.loads(arguments or "{}")
            if name not in self.cfg.agent_tools:
                return f"Công cụ không tồn tại: {name}"
            if name == "kg_related":
                from kg import render_related

                hops = min(MAX_HOPS, max(1, int(a.get("hops", 1))))
                return render_related(str(a["law"]), self.tools.related(str(a["law"]), hops))
            if name == "web_search":
                rs = self.tools.web.search(str(a["query"]))
                return "\n".join(f"- {r['title'][:100]} ({r['url']}): {r['snippet'][:WEB_SNIPPET_CHARS]}" for r in rs) or "Không tìm thấy kết quả trên web."
            if name == "search_law":
                ps = self.tools.search_law(str(a["query"]))
                return "\n\n".join(f"[{i}] ({p.doc})\n{p.text[:RESULT_CHARS]}" for i, p in enumerate(ps, 1)) or "Không tìm thấy."
            if name == "get_article":
                return (self.tools.article(a["law"], int(a["dieu"])) or "Không tìm thấy Điều này.")[:ARTICLE_CHARS]
            return f"Công cụ không tồn tại: {name}"
        except Exception as e:  # noqa: BLE001 — a tool error goes back to the model as text
            return f"Lỗi công cụ: {repr(e)[:200]}"
