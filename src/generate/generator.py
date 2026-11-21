"""Send a prompt to the LLM and clean the answer.

The LLM is LlamaIndex `OpenAILike`, so the same code talks to vLLM on the L40S box, mlx_lm on the Mac or Modal:
only `backend.base_url` changes (configs/backends/*.yaml).

Decoding (configs `generate:`; research in docs/notes/09-inference-configs):
  thinking off / None   one chat call; top_k, min_p, penalties and `enable_thinking` go in vLLM's `extra_body`
  thinking on           datamate pattern: render the chat template client-side (enable_thinking=True), complete
                        the thought up to `think_tokens` (stop at </think>), force "</think>", complete the answer
  mbr: [t1, t2, …]      one sample per temperature, keep the one with the highest mean ROUGE-L to the others
"""
import asyncio
import re

from config import Backend, Generate
from generate.check import Gate
from generate.prompt import Prompt
from records import Passage

EXTRA = ("top_k", "min_p", "presence_penalty", "repetition_penalty")


class Generator:
    def __init__(self, cfg: Generate, backend: Backend):
        self.cfg, self.backend = cfg, backend
        self.prompt = Prompt.load(cfg.prompt)
        self._llm = self._completer = self._tokenizer = self._shots = None

    @property
    def llm(self):
        if self._llm is None:  # built on first use: constructing the client is cheap, but keep __init__ trivial
            self._llm = self._client(chat=self.cfg.mode == "chat", max_tokens=self.cfg.max_tokens)
        return self._llm

    @property
    def completer(self):
        """/v1/completions client without a fixed max_tokens (set per call) — for the thinking path."""
        if self._completer is None:
            self._completer = self._client(chat=False, max_tokens=None)
        return self._completer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        return self._tokenizer

    def _client(self, chat: bool, max_tokens: int | None):
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(model=self.cfg.model, api_base=self.backend.base_url, api_key=self.backend.api_key,
                          is_chat_model=chat, temperature=self.cfg.temperature, max_tokens=max_tokens,
                          timeout=600, max_retries=2, context_window=32768)

    def sampling(self, temperature: float | None = None, thinking_switch: bool = True) -> dict:
        """Per-call kwargs: temperature, top_p, and vLLM-only fields in extra_body."""
        g = self.cfg
        kw = {"temperature": g.temperature if temperature is None else temperature}
        if g.top_p is not None:
            kw["top_p"] = g.top_p
        extra = {k: getattr(g, k) for k in EXTRA if getattr(g, k) is not None}
        if thinking_switch and g.thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": g.thinking}
        if extra:
            kw["extra_body"] = extra
        return kw

    def shots(self, question: str) -> list[tuple[str, str]]:
        if not self.cfg.shots:
            return []
        if self._shots is None:
            from generate.shots import ShotStore

            self._shots = ShotStore()
        return self._shots.pick(question, self.cfg.shots)

    async def answer(self, question: str, passages: list[Passage]) -> str:
        if not self.cfg.mbr:
            return await self.sample(question, passages)
        cands = await self.candidates(question, passages)
        if self.cfg.check:   # optional gate: keep candidates that pass; if none does, keep them all (never end up with nothing)
            gate = Gate(self.cfg.check)
            cands = [c for c in cands if gate.passes(c, passages)] or cands
        return self.mbr_pick(cands)

    async def candidates(self, question: str, passages: list[Passage]) -> list[str]:
        """One sample per MBR temperature, before any gate or pick (also used to measure the best-of-n ceiling)."""
        return list(await asyncio.gather(*(self.sample(question, passages, t) for t in self.cfg.mbr)))

    async def sample(self, question: str, passages: list[Passage], temperature: float | None = None) -> str:
        if self.cfg.mode != "chat":
            text = (await self.llm.acomplete(self.prompt.completion(question, passages, self.shots(question)), **self.sampling(temperature))).text
        elif self.cfg.thinking:
            text = await self._think(self.prompt.messages(question, passages, self.shots(question)), temperature)
        else:
            from llama_index.core.llms import ChatMessage

            msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in self.prompt.messages(question, passages, self.shots(question))]
            text = (await self.llm.achat(msgs, **self.sampling(temperature))).message.content or ""
        return self.clean(text)

    async def stream(self, question: str, passages: list[Passage]):
        """Text deltas of one greedy chat call (the deployed path: no MBR, no thinking). Reasoning blocks are not expected here."""
        from llama_index.core.llms import ChatMessage

        msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in self.prompt.messages(question, passages, self.shots(question))]
        async for chunk in await self.llm.astream_chat(msgs, **self.sampling()):
            if chunk.delta:
                yield chunk.delta

    async def _think(self, messages: list[dict], temperature: float | None) -> str:
        """Thought capped at think_tokens, then the answer after a forced </think>."""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=True)
        if not prompt.rstrip().endswith("<think>"):   # Qwen3 leaves <think> to the model, Qwen3.5 opens it
            prompt += "<think>\n"
        kw = self.sampling(temperature, thinking_switch=False)
        thought = (await self.completer.acomplete(prompt, max_tokens=self.cfg.think_tokens, stop=["</think>"], **kw)).text
        prompt += thought.rstrip() + "\n</think>\n\n"
        return (await self.completer.acomplete(prompt, max_tokens=self.cfg.max_tokens, **kw)).text

    @staticmethod
    def mbr_pick(cands: list[str]) -> str:
        """Minimum Bayes risk with ROUGE-L (the course metric) as utility: the most central candidate wins."""
        from evaluate.metrics import rouge_l_score

        cands = [c for c in cands if c] or [""]
        if len(cands) == 1:
            return cands[0]
        score = [sum(rouge_l_score(a, b) for j, b in enumerate(cands) if j != i) for i, a in enumerate(cands)]
        return cands[max(range(len(cands)), key=score.__getitem__)]

    @staticmethod
    def clean(text: str) -> str:
        """Drop reasoning blocks (Qwen3 <think>, also an unclosed one at the start) and surrounding whitespace."""
        text = re.sub(r"(?s)<think>.*?</think>", "", text)
        return re.sub(r"(?s)^.*?</think>", "", text).strip()


class FakeGenerator(Generator):
    """No server: copies the top passage. Lets the whole pipeline run in tests and smoke runs."""

    async def answer(self, question: str, passages: list[Passage]) -> str:
        return passages[0].text if passages else ""
