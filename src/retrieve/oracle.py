"""Oracle mode: put a gold passage (from the dense candidate pool) at rank 1 when the top-k has none."""
from typing import Callable

from records import Passage


def inject_gold(top: list[Passage], pool: list[Passage], is_gold: Callable[[Passage], bool]) -> list[Passage]:
    if not top or any(is_gold(p) for p in top):
        return list(top)
    gold = next((p for p in pool if is_gold(p)), None)
    return list(top) if gold is None else [gold, *top[:-1]]
