import json

from train.react.react_sft import assistant_spans


def render(msgs, add_generation_prompt=False):
    """A deterministic stand-in for the chat template."""
    out = ""
    for m in msgs:
        body = m.get("content") or ""
        for c in m.get("tool_calls") or []:
            body += f"<tool_call>{c['function']['name']}</tool_call>"
        out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def convo(*turns):
    return [{"role": r, "content": c, **({"tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_article", "arguments": "{}"}}]} if r == "assistant" and c == "" else {})}
            for r, c in turns]


def spans_text(msgs):
    full = render(msgs)
    return [full[s:e] for s, e in assistant_spans(msgs, render)]


def test_only_the_assistant_turn_is_a_span_and_it_includes_its_closing_im_end():
    msgs = convo(("system", "SYS"), ("user", "QUESTION"), ("assistant", "ANSWER"))
    assert spans_text(msgs) == ["ANSWER<|im_end|>"]


def test_each_assistant_turn_gets_a_span_and_tool_results_stay_masked():
    msgs = convo(("system", "SYS"), ("user", "Q"), ("assistant", ""), ("tool", "RESULT"), ("assistant", "FINAL"))
    assert spans_text(msgs) == ["<tool_call>get_article</tool_call><|im_end|>", "FINAL<|im_end|>"]


def test_a_template_that_rewrites_earlier_turns_is_rejected_instead_of_mislabelling():
    import pytest

    def unstable(msgs, add_generation_prompt=False):
        text = render(msgs, add_generation_prompt)
        return text.replace("FIRST", "FIRST<think></think>") if len(msgs) > 4 else text   # earlier assistant turn changes once more turns follow

    msgs = convo(("system", "S"), ("user", "Q"), ("assistant", "FIRST"), ("user", "Q2"), ("assistant", "SECOND"))
    with pytest.raises(ValueError):
        assistant_spans(msgs, unstable)


class FakeTokenizer:
    """Same interface subset as a HF fast tokenizer: apply_chat_template + __call__ with offsets. Tokens = whitespace-delimited words."""

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        assert tokenize is False and enable_thinking is False
        self.seen.append((messages, tools))
        return render(messages, add_generation_prompt)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re

        words = list(re.finditer(r"\S+", text))
        return {"input_ids": list(range(1, len(words) + 1)), "offset_mapping": [m.span() for m in words]}


def test_build_examples_labels_only_assistant_tokens_and_parses_tool_arguments_for_the_template():
    from train.react.react_sft import build_examples

    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_article", "arguments": json.dumps({"dieu": 5})}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "RESULT"}, {"role": "assistant", "content": "FINAL"}]
    tok = FakeTokenizer()
    [ex] = build_examples([{"qid": "q1", "messages": msgs, "tools": [{"type": "function"}]}], tok)
    words = render(msgs).split()
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"]) == len(words)
    trained = [w for w, lab in zip(words, ex["labels"]) if lab != -100]
    assert trained == ["<tool_call>get_article</tool_call><|im_end|>", "FINAL<|im_end|>"]
    assert all(lab == tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100)
    sent, tools = tok.seen[-1]
    assert tools == [{"type": "function"}]
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == {"dieu": 5}      # Qwen's template iterates arguments as a mapping
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"dieu": 5})   # the input rows are not mutated


def test_build_examples_skips_rows_longer_than_max_length_instead_of_truncating():
    from train.react.react_sft import build_examples

    short = {"qid": "a", "messages": convo(("system", "S"), ("user", "Q"), ("assistant", "A")), "tools": []}
    long = {"qid": "b", "messages": convo(("system", "S"), ("user", "Q " * 50), ("assistant", "A")), "tools": []}
    assert len(build_examples([short, long], FakeTokenizer())) == 2
    kept = build_examples([short, long], FakeTokenizer(), max_length=10)
    assert len(kept) == 1 and len(kept[0]["input_ids"]) <= 10
