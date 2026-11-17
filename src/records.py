"""The records passed between stages. Every stage reads and writes only these."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Question:
    qid: int
    text: str
    answer: str             # reference answer (RES sheet)
    law_id: str = ""        # cited document number, e.g. 50/2024/TT-NHNN


@dataclass
class Passage:
    pid: str                # node id in the index
    doc: str                # corpus file stem = law id with "/" -> "_"
    text: str
    score: float = 0.0


@dataclass
class Prediction:
    qid: int
    answer: str
    passages: list[str] = field(default_factory=list)   # doc of each passage given to the LLM, in order
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
