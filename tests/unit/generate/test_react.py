import asyncio
from types import SimpleNamespace as NS

from config import Backend, Generate
from generate.react import ReactGenerator
from records import Passage


def call(name, args, id_="c1"):
    return NS(id=id_, function=NS(name=name, arguments=args))


class FakeOAI:
    """chat.completions.create replays scripted messages and records the kwargs it saw."""

    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kw):
        self.seen.append(kw)
        m = self.script.pop(0)
        return NS(choices=[NS(message=m)])


class FakeTools:
    def __init__(self):
        self.calls = []

    def search_law(self, q):
        self.calls.append(("search", q))
        return [Passage("p", "50_2024_TT", "Điều 5. Nội dung", 1.0)]

    def article(self, law, dieu):
        self.calls.append(("article", law, dieu))
        return "Điều 5. Toàn văn"


def gen(script, max_turns=4):
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", max_turns=max_turns, agent=True)
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI(script)
    return g


def run(g, q="câu hỏi?"):
    return asyncio.run(g.answer(q, [Passage("p0", "a", "ngữ cảnh", 1.0)]))


def test_answer_without_tool_call_is_final():
    g = gen([NS(content="Căn cứ Điều 5.", tool_calls=None)])
    assert run(g) == "Căn cứ Điều 5."
    assert g._oai.seen[0]["tools"]            # tools offered on the first turn


def test_tool_call_then_answer():
    g = gen([NS(content="", tool_calls=[call("get_article", '{"law": "50/2024/TT", "dieu": 5}')]),
             NS(content="Xong.", tool_calls=None)])
    assert run(g) == "Xong."
    assert g.tools.calls == [("article", "50/2024/TT", 5)]
    last = g._oai.seen[1]["messages"][-1]
    assert last["role"] == "tool" and "Toàn văn" in last["content"]


def test_repeated_call_is_not_run_twice_and_last_turn_has_no_tools():
    same = lambda i: NS(content="", tool_calls=[call("search_law", '{"query": "x"}', f"c{i}")])
    g = gen([same(1), same(2), NS(content="Trả lời.", tool_calls=None)], max_turns=3)
    assert run(g) == "Trả lời."
    assert g.tools.calls == [("search", "x")]                 # second identical call skipped
    assert "tools" not in g._oai.seen[2]                      # last turn: tools removed
    assert "ngay bây giờ" in g._oai.seen[2]["messages"][-1]["content"]


def test_tool_error_goes_back_as_text():
    g = gen([NS(content="", tool_calls=[call("get_article", "not json")]), NS(content="ok", tool_calls=None)])
    assert run(g) == "ok"
    assert "Lỗi công cụ" in g._oai.seen[1]["messages"][-1]["content"]


def test_seed_is_stable_per_question():
    a, b = gen([NS(content="x", tool_calls=None)]), gen([NS(content="x", tool_calls=None)])
    run(a), run(b)
    assert a._oai.seen[0]["seed"] == b._oai.seen[0]["seed"]


def names(g):
    return [t["function"]["name"] for t in g.tool_specs()]


def test_tool_set_comes_from_the_config():
    base = Generate(model="m", params_b=2.0, agent=True)
    assert names(ReactGenerator(base, Backend(name="x", base_url="http://x/v1"))) == ["search_law", "get_article"]
    kg = base.model_copy(update={"agent_tools": ["get_article", "kg_related"]})
    assert names(ReactGenerator(kg, Backend(name="x", base_url="http://x/v1"))) == ["get_article", "kg_related"]


def test_kg_related_description_advises_a_small_number_of_hops():
    g = ReactGenerator(Generate(model="m", params_b=2.0, agent=True, agent_tools=["kg_related"]), Backend(name="x", base_url="http://x/v1"))
    d = g.tool_specs()[0]["function"]
    assert "hops" in d["parameters"]["properties"] and "1" in d["description"] and "2" in d["description"]   # advice: usually 1-2


def test_kg_related_tool_returns_hop_counts_and_caps_the_size():
    class KGTools(FakeTools):
        def related(self, law, hops):
            self.calls.append(("related", law, hops))
            return [{"hop": 1, "total": 40, "docs": [{"id": "1", "key": "97/2025/QH15", "family": "amends", "title": "T", "status": "Còn hiệu lực"}]}]

    g = ReactGenerator(Generate(model="m", params_b=2.0, agent=True, agent_tools=["kg_related"]), Backend(name="x", base_url="http://x/v1"), KGTools())
    out = g.run_tool("kg_related", '{"law": "75/2015/QH13", "hops": 99}')
    assert "hop 1: 40 văn bản, hiện 1" in out
    assert g.tools.calls == [("related", "75/2015/QH13", 4)]     # an absurd hop count is clamped (MAX_HOPS), never explored


def test_a_tool_not_in_the_config_is_refused():
    g = gen([NS(content="x", tool_calls=None)])
    assert "không tồn tại" in g.run_tool("kg_related", '{"law": "a"}')


def system_text(tools):
    g = ReactGenerator(Generate(model="m", params_b=2.0, prompt="vi_rag", agent=True, agent_tools=tools), Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI([NS(content="x", tool_calls=None)])
    run(g)
    return g._oai.seen[0]["messages"][0]["content"]


def test_system_prompt_lists_only_enabled_tools_and_advises_hops_for_the_kg_tool():
    base = system_text(["search_law", "get_article"])
    assert "search_law" in base and "get_article" in base and "kg_related" not in base
    kg = system_text(["get_article", "kg_related"])
    assert "kg_related" in kg and "search_law(" not in kg and "hops" in kg


class FakeWeb:
    def __init__(self, results=None, fail=None):
        self.results, self.fail, self.queries = results or [], fail, []

    def search(self, q):
        self.queries.append(q)
        if self.fail:
            raise self.fail
        return self.results


def web_gen(web, tools=("web_search",)):
    cfg = Generate(model="m", params_b=2.0, agent=True, agent_tools=list(tools))
    t = FakeTools()
    t.web = web
    return ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), t)


def test_web_search_tool_lists_title_url_and_a_cut_snippet():
    g = web_gen(FakeWeb([{"url": "https://vbpl.vn/a", "title": "Nghị định 94/2025", "snippet": "x" * 500}]))
    out = g.run_tool("web_search", '{"query": "nghị định 94/2025"}')
    assert "Nghị định 94/2025" in out and "https://vbpl.vn/a" in out and len(out) < 450
    assert g.tools.web.queries == ["nghị định 94/2025"]


def test_web_search_with_no_result_or_a_service_error_goes_back_as_text():
    assert "Không tìm thấy" in web_gen(FakeWeb([])).run_tool("web_search", '{"query": "q"}')
    out = web_gen(FakeWeb(fail=RuntimeError("401 unauthorized"))).run_tool("web_search", '{"query": "q"}')
    assert "Lỗi công cụ" in out and "401" in out


def test_web_search_is_refused_when_it_is_not_in_the_tool_set():
    assert "không tồn tại" in web_gen(FakeWeb([R_]), tools=("get_article",)).run_tool("web_search", '{"query": "q"}')


R_ = {"url": "https://x.example", "title": "t", "snippet": "s"}


def test_system_prompt_mentions_web_search_only_when_enabled():
    assert "web_search" not in system_text(["search_law"]) and "web_search" in system_text(["search_law", "web_search"])


def gated(script, check, max_turns=4):
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", agent=True, max_turns=max_turns, check=check)
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI(script)
    return g, asyncio.run(g.answer("câu hỏi?", [Passage("p0", "a", "Điều 5. ngữ cảnh", 1.0)]))


def test_a_failing_answer_gets_exactly_one_repair_nudge_and_the_second_answer_is_final():
    g, out = gated([NS(content="Căn cứ Điều 99 quy định.", tool_calls=None), NS(content="Căn cứ Điều 5 quy định.", tool_calls=None)], ["grounded_citation"])
    assert out == "Căn cứ Điều 5 quy định."
    nudge = g._oai.seen[1]["messages"][-1]
    assert nudge["role"] == "user" and "Điều 99" in nudge["content"]


def test_a_second_failing_answer_is_accepted_not_looped():
    g, out = gated([NS(content="Căn cứ Điều 99.", tool_calls=None), NS(content="Vẫn Điều 98.", tool_calls=None), NS(content="never", tool_calls=None)], ["grounded_citation"])
    assert out == "Vẫn Điều 98." and len(g._oai.seen) == 2


def test_without_a_gate_the_first_answer_is_final_and_tool_results_count_as_context():
    _, out = gated([NS(content="Căn cứ Điều 99.", tool_calls=None)], [])
    assert out == "Căn cứ Điều 99."
    g, out = gated([NS(content="", tool_calls=[call("get_article", '{"law": "a", "dieu": 5}')]), NS(content="Căn cứ Điều 5 quy định.", tool_calls=None)], ["grounded_citation"])
    assert out == "Căn cứ Điều 5 quy định." and len(g._oai.seen) == 2                # Điều 5 is in the initial passage: no repair


def test_the_full_conversation_of_each_question_is_kept_for_trajectory_training():
    g = gen([NS(content="", tool_calls=[call("get_article", '{"law": "50/2024/TT", "dieu": 5}')]), NS(content="Xong.", tool_calls=None)])
    run(g, "câu hỏi?")
    msgs = g.traces["câu hỏi?"]
    assert [m["role"] for m in msgs][-3:] == ["assistant", "tool", "assistant"]
    assert msgs[-3]["tool_calls"][0]["function"]["name"] == "get_article" and msgs[-1]["content"] == "Xong."


def test_compact_tool_prompt_drops_the_duplicated_tool_list_but_keeps_the_answer_template():
    from generate.react import system_tools

    full, compact = system_tools(["search_law", "get_article"]), system_tools(["search_law", "get_article"], "compact")
    assert "- search_law(" in full and "- search_law(" not in compact and "- get_article(" not in compact
    assert "Căn cứ Điều X" in compact and len(compact) < len(full) * 0.6


def test_compact_prompt_keeps_the_advice_that_only_the_prose_carried():
    from generate.react import system_tools

    assert "web_search" in system_tools(["search_law", "web_search"], "compact")         # when to use the web is not in the JSON schema
    assert "web_search" not in system_tools(["search_law", "get_article"], "compact")


def test_the_style_is_chosen_by_the_config_and_reaches_the_first_message():
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", agent=True, tool_prompt="compact")
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI([NS(content="ok", tool_calls=None)])
    run(g)
    system = g._oai.seen[0]["messages"][0]["content"]
    assert "Căn cứ Điều X" in system and "- search_law(" not in system


def seed_of(agent_seed, q="câu hỏi?"):
    g = gen([NS(content="ok", tool_calls=None)])
    g.cfg = g.cfg.model_copy(update={"agent_seed": agent_seed})
    run(g, q)
    return g._oai.seen[0]["seed"]


def test_agent_seed_zero_keeps_the_old_per_question_seed():
    import zlib
    assert seed_of(0) == zlib.crc32("câu hỏi?".encode()) & 0x7FFFFFFF


def test_agent_seed_changes_the_per_question_seed_deterministically():
    assert seed_of(1) == seed_of(1)
    assert seed_of(1) != seed_of(2) and seed_of(1) != seed_of(0)


def test_force_prompt_requires_calling_the_tools_before_answering_and_has_no_only_when_insufficient_rule():
    from generate.react import system_tools
    force = system_tools(["search_law", "get_article"], "force")
    assert "search_law" in force and "Căn cứ Điều X" in force
    assert "Chỉ gọi công cụ khi" not in force and "chưa đủ" not in force


def test_min_tool_calls_nudges_a_model_that_answers_without_calling_a_tool():
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", max_turns=4, agent=True, min_tool_calls=1)
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI([NS(content="Trả lời vội.", tool_calls=None),
                      NS(content="", tool_calls=[call("search_law", '{"query": "q"}')]),
                      NS(content="Căn cứ Điều 5.", tool_calls=None)])
    assert run(g) == "Căn cứ Điều 5."
    assert g.tools.calls == [("search", "q")]
    nudges = [m for m in g._oai.seen[-1]["messages"] if m["role"] == "user" and "chưa gọi công cụ" in m["content"]]
    assert len(nudges) == 1 and "search_law" in nudges[0]["content"]             # the reminder is in the conversation
    assert g.traces["câu hỏi?"][-2]["role"] == "tool"                          # the rushed first answer is not the final answer


def test_min_tool_calls_zero_keeps_the_old_behaviour():
    g = gen([NS(content="Căn cứ Điều 5.", tool_calls=None)])
    assert run(g) == "Căn cứ Điều 5." and len(g._oai.seen) == 1


def test_without_initial_passages_the_user_message_is_just_the_question_no_refusal_rule():
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", max_turns=3, agent=True, agent_initial_k=0)
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI([NS(content="Căn cứ Điều 5.", tool_calls=None)])
    run(g, "Mức phạt là bao nhiêu?")
    user = [m for m in g._oai.seen[0]["messages"] if m["role"] == "user"][0]["content"]
    assert "Mức phạt là bao nhiêu?" in user
    assert "từ chối" not in user and "Ngữ cảnh" not in user     # the RAG rule would make the model refuse an empty context instead of searching


def test_concurrent_first_tool_calls_build_the_index_only_once():
    import threading
    import time

    from generate.react import LawTools

    built = []

    class Slow(LawTools):
        def _build(self):
            built.append(1)
            time.sleep(0.2)                      # a real build reads a ~10 GB docstore: concurrent builds exhausted the RAM (OOM, exit 137)
            self._retriever, self._reranker = object(), None
            self.store = NS(nodes=lambda: [])

    t = Slow(cfg=None)
    ts = [threading.Thread(target=t.article, args=("50/2024/TT", 5)) for _ in range(6)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    assert len(built) == 1


def test_search_with_question_mixes_the_questions_own_hits_into_every_search_result():
    class T(FakeTools):
        def search_law(self, q):
            self.calls.append(("search", q))
            return [Passage(f"{q}-{i}", "d", f"{q} text {i}", 1.0) for i in range(5)]

    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", max_turns=3, agent=True, search_with_question=True)
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), T())
    g._oai = FakeOAI([NS(content="", tool_calls=[call("search_law", '{"query": "truy vấn riêng"}')]), NS(content="Căn cứ Điều 5.", tool_calls=None)])
    run(g, "câu hỏi gốc?")
    assert ("search", "câu hỏi gốc?") in g.tools.calls and ("search", "truy vấn riêng") in g.tools.calls
    tool_msg = [m for m in g._oai.seen[-1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "câu hỏi gốc? text 0" in tool_msg and "truy vấn riêng text 0" in tool_msg    # both sources present
    assert tool_msg.count("[") == 5                                                    # still k results


def test_profile_goes_into_the_system_message_when_enabled():
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", agent=True, agent_profile=True)
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI([NS(content="Căn cứ Điều 5.", tool_calls=None)])
    asyncio.run(g.answer("Điều 5 Thông tư 23/2014/TT-NHNN quy định gì?", [Passage("p0", "23_2014_TT-NHNN", "Quy định\nĐiều 4. Tiêu đề\nnội dung", 1.0)]))
    system = g._oai.seen[0]["messages"][0]["content"]
    assert "[HỒ SƠ TRUY XUẤT]" in system and "Điều 5 của 23/2014/TT-NHNN chưa có" in system


def test_profile_is_off_by_default():
    g = gen([NS(content="Căn cứ Điều 5.", tool_calls=None)])
    run(g)
    assert "HỒ SƠ" not in str(g._oai.seen[0]["messages"])


def test_agent_with_no_tools_sends_no_tools_parameter():
    cfg = Generate(model="m", params_b=2.0, prompt="vi_rag", agent=True, agent_tools=[])
    g = ReactGenerator(cfg, Backend(name="x", base_url="http://x/v1"), FakeTools())
    g._oai = FakeOAI([NS(content="Căn cứ Điều 5.", tool_calls=None)])
    assert run(g) == "Căn cứ Điều 5."
    assert "tools" not in g._oai.seen[0]
