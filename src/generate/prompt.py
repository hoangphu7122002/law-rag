"""Prompt files: generate/prompts/<name>.md

    ---
    context: plain | numbered | ngu_canh   # how passages are joined into {context}
    ---
    ## system                      (optional)
    ...
    ## user
    ... {context} ... {question} ...

The same Prompt is used for inference and for building SFT data, so the two formats cannot drift.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from records import Passage

FOLDER = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    user: str
    system: str = ""
    context_style: str = "plain"

    @classmethod
    def load(cls, name: str) -> "Prompt":
        text = (FOLDER / f"{name}.md").read_text(encoding="utf-8")
        meta = {}
        if text.startswith("---"):
            _, front, text = text.split("---", 2)
            meta = yaml.safe_load(front) or {}
        sections = dict(re.findall(r"(?ms)^## (system|user)\n(.*?)(?=^## |\Z)", text))
        return cls(name=name, user=sections["user"].strip("\n"), system=sections.get("system", "").strip(),
                   context_style=meta.get("context", "plain"))

    def context(self, passages: list[Passage]) -> str:
        if self.context_style == "ngu_canh":  # AITeamVN Vi-Qwen2-RAG card: "- Ngữ cảnh 1: ..."
            return "\n\n".join(f"- Ngữ cảnh {i}: {p.text}" for i, p in enumerate(passages, 1))
        if self.context_style == "numbered":
            return "\n\n".join(f"[{i}] ({p.doc})\n{p.text}" for i, p in enumerate(passages, 1))
        return "\n".join(p.text for p in passages)

    def messages(self, question: str, passages: list[Passage], shots: list[tuple[str, str]] | None = None) -> list[dict]:
        user = self.user.format(context=self.context(passages), question=question)
        if shots:   # few-shot: nearest train question/answer pairs (generate/shots.py), no context, before the task
            user = "Ví dụ câu hỏi và cách trả lời tương tự:\n" + "\n\n".join(
                f"### Câu hỏi : {q}\n### Trả lời : {a}" for q, a in shots) + "\n\n" + user
        return ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": user}]

    def completion(self, question: str, passages: list[Passage], shots: list[tuple[str, str]] | None = None) -> str:
        """For base (non-chat) models: system text, then the user text, ending in a newline like the notebook."""
        return "\n\n".join(m["content"] for m in self.messages(question, passages, shots)) + "\n"
