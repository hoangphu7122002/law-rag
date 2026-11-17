"""Question + passages -> answer.

Prompt      a prompt file in generate/prompts/<name>.md -> chat messages or one completion string
Generator   sends it to the LLM (any OpenAI-compatible server through LlamaIndex OpenAILike), cleans the answer
FakeGenerator  no server: answers with the top passage (tests, smoke runs)
"""
from generate.generator import FakeGenerator, Generator
from generate.prompt import Prompt

__all__ = ["FakeGenerator", "Generator", "Prompt"]
