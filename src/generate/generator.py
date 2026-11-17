"""Send a prompt to the LLM and clean the answer.

The LLM is LlamaIndex `OpenAILike`, so the same code talks to vLLM on the L40S box, mlx_lm on the Mac or Modal:
only `backend.base_url` changes (configs/backends/*.yaml).
"""
import re

from config import Backend, Generate
from generate.prompt import Prompt
from records import Passage


class Generator:
    def __init__(self, cfg: Generate, backend: Backend):
        self.cfg, self.backend = cfg, backend
        self.prompt = Prompt.load(cfg.prompt)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:  # built on first use: constructing the client is cheap, but keep __init__ trivial
            from llama_index.llms.openai_like import OpenAILike

            self._llm = OpenAILike(model=self.cfg.model, api_base=self.backend.base_url, api_key=self.backend.api_key,
                                   is_chat_model=self.cfg.mode == "chat", temperature=self.cfg.temperature,
                                   max_tokens=self.cfg.max_tokens, timeout=600, max_retries=2, context_window=32768)
        return self._llm

    async def answer(self, question: str, passages: list[Passage]) -> str:
        if self.cfg.mode == "chat":
            from llama_index.core.llms import ChatMessage

            msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in self.prompt.messages(question, passages)]
            text = (await self.llm.achat(msgs)).message.content or ""
        else:
            text = (await self.llm.acomplete(self.prompt.completion(question, passages))).text
        return self.clean(text)

    @staticmethod
    def clean(text: str) -> str:
        """Drop reasoning blocks (Qwen3 <think>) and surrounding whitespace."""
        return re.sub(r"(?s)<think>.*?</think>", "", text).strip()


class FakeGenerator(Generator):
    """No server: copies the top passage. Lets the whole pipeline run in tests and smoke runs."""

    async def answer(self, question: str, passages: list[Passage]) -> str:
        return passages[0].text if passages else ""
