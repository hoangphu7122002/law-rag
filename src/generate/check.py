"""Optional answer gate (config `generate.check: [..]`, empty = off). A gate says pass / fail for one answer given the context the model saw.

Why code checks and not abstain: our metric punishes refusals (template rougeL .066), so a failed gate means "pick another candidate / ask once more", never "refuse".
Calibration rule borrowed from datamate: a check that fires on more than 5 % of CORRECT answers must stay off (measure on train gold before enabling).
"""
import re

from records import Passage

DIEU = re.compile(r"Điều\s+(\d+[a-zđ]?)", re.I)


def grounded_citation(answer: str, passages: list[Passage]) -> bool:
    ctx = " ".join(p.text for p in passages)
    return all(re.search(rf"Điều\s+{re.escape(n)}\b", ctx, re.I) for n in set(DIEU.findall(answer)))


CHECKS = {"grounded_citation": grounded_citation}


class Gate:
    def __init__(self, names: list[str]):
        unknown = [n for n in names if n not in CHECKS]
        if unknown:
            raise ValueError(f"unknown check: {unknown[0]}")
        self.names = names

    def passes(self, answer: str, passages: list[Passage]) -> bool:
        return all(CHECKS[n](answer, passages) for n in self.names)
