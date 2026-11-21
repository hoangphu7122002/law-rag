import asyncio
from types import SimpleNamespace

from config import Backend, Generate
from generate.generator import Generator

BACKEND = Backend(name="t", base_url="http://x/v1", api_key="k")


def gen(**kw) -> Generator:
    return Generator(Generate(model="m", params_b=1.7, prompt="vi_rag", **kw), BACKEND)


def test_sampling_defaults_send_only_temperature():
    assert gen().sampling() == {"temperature": 0.0}


def test_sampling_extra_body_and_thinking_switch():
    g = gen(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, thinking=False)
    kw = g.sampling(0.3)
    assert kw["temperature"] == 0.3 and kw["top_p"] == 0.8
    assert kw["extra_body"] == {"top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}}
    assert "chat_template_kwargs" not in g.sampling(thinking_switch=False)["extra_body"]


def test_mbr_pick_most_central():
    assert Generator.mbr_pick(["a b c d", "a b c e", "x y z"]) in ("a b c d", "a b c e")
    assert Generator.mbr_pick(["", "only"]) == "only"


def test_clean_strips_think():
    assert Generator.clean("<think>hmm</think>\n Đáp án") == "Đáp án"
    assert Generator.clean("unclosed thought</think>Đáp") == "Đáp"


def test_think_path_caps_thought_then_forces_close():
    g = gen(thinking=True, think_tokens=7, max_tokens=11)
    calls = []

    class Fake:
        async def acomplete(self, prompt, **kw):
            calls.append((prompt, kw))
            return SimpleNamespace(text="thought" if "stop" in kw else "answer")

    g._completer = Fake()
    g._tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "<|im_start|>assistant\n")
    out = asyncio.run(g._think([{"role": "user", "content": "q"}], None))
    assert out == "answer"
    assert calls[0][1]["max_tokens"] == 7 and calls[0][1]["stop"] == ["</think>"]
    assert calls[1][0].endswith("<think>\nthought\n</think>\n\n") and calls[1][1]["max_tokens"] == 11


def test_run_key_unchanged_by_new_defaults():
    from config import load

    a = load("configs/exp/A2e_dedup.yaml", [], None)
    b = load("configs/exp/A2e_dedup.yaml", ["generate.top_p=0.8"], None)
    assert a.run_key() != b.run_key()


def run_mbr(check, answers):
    from records import Passage

    g = gen(mbr=[0.0, 0.7, 0.7, 0.7], check=check)
    it = iter(answers)

    async def fake_sample(q, passages, temperature=None):
        return next(it)

    g.sample = fake_sample
    return asyncio.run(g.answer("câu hỏi", [Passage("p", "d", "Điều 5. Nội dung", 1.0)]))


def test_gate_drops_ungrounded_candidates_before_the_consensus_pick():
    wrong = "Căn cứ Điều 99 quy định về phạm vi áp dụng của nghị định"          # two identical wrong answers are the most "central"
    right = "Căn cứ Điều 5 quy định về phạm vi áp dụng của nghị định này"
    assert run_mbr([], [wrong, wrong, wrong, right]) == wrong
    assert run_mbr(["grounded_citation"], [wrong, wrong, wrong, right]) == right


def test_when_every_candidate_fails_the_gate_the_pick_falls_back_to_all_of_them():
    bad = "Căn cứ Điều 99 quy định"
    assert run_mbr(["grounded_citation"], [bad, bad, bad, bad]) == bad


def test_candidates_returns_every_sample_and_answer_picks_among_them():
    from records import Passage

    g = gen(mbr=[0.0, 0.7, 0.7])
    outs = iter(["a b c d", "a b c e", "x y z"])

    async def fake_sample(q, passages, temperature=None):
        return next(outs)

    g.sample = fake_sample
    cands = asyncio.run(g.candidates("q", [Passage("p", "d", "Điều 5. Nội dung", 1.0)]))
    assert cands == ["a b c d", "a b c e", "x y z"]


def test_stream_yields_text_deltas_of_a_chat_call():
    import asyncio
    from types import SimpleNamespace

    from records import Passage

    class FakeLLM:
        async def astream_chat(self, msgs, **kw):
            async def gen():
                for d in ["Căn cứ ", "tại ", "Điều 5."]:
                    yield SimpleNamespace(delta=d)
            return gen()

    g = gen(mode="chat")
    g._llm = FakeLLM()

    async def run():
        return [d async for d in g.stream("q?", [Passage(pid="1", doc="A", text="Điều 5. x", score=1.0)])]

    assert asyncio.run(run()) == ["Căn cứ ", "tại ", "Điều 5."]
