"""SFT / RAFT data: train questions (decontaminated) + their retrieved passages, rendered with the SAME prompt and chat
template as inference, target = the reference answer. Thinking models are trained in non-thinking mode (the
references have no reasoning), i.e. the template's empty <think></think> block precedes the answer.

    SFTData(cfg, "q35_v0").build()   ->  data/train/q35_v0/{train.jsonl, val.jsonl, report.json}
    val = decontaminated dev (loss curve / best checkpoint, as datamate); none when dev is trained on (final model)
    rows: {"qid", "prompt": <rendered chat prompt>, "completion": <answer><|im_end|>}
"""
import json
import random
from pathlib import Path

from config import Config
from corpus.decontam import Decontam
from corpus.qa import QASet
from generate.prompt import Prompt
from evaluate.retrieval_eval import RetrievalEval
from generate.shots import ShotStore
from retrieve import Retrieval

OUT = Path("data/train")
REFUSALS = [
    "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan để trả lời câu hỏi này.",
    "Các ngữ cảnh được cung cấp không đề cập đến nội dung mà câu hỏi nêu ra, nên chưa thể trả lời câu hỏi này.",
    "Ngữ cảnh được cung cấp không có thông tin về vấn đề này. Vui lòng cung cấp thêm văn bản quy phạm pháp luật liên quan để được trả lời chính xác.",
    "Trong các ngữ cảnh được cung cấp không có quy định nào liên quan đến câu hỏi \"{q}\", vì vậy chưa thể đưa ra câu trả lời.",
]


class SFTData:
    def __init__(self, cfg: Config, name: str, splits: tuple[str, ...] = ("train",),
                 k_range: tuple[int, int] | None = None, oracle: float = 0.0, shots: tuple[int, ...] = (0,),
                 refuse: float = 0.0, seed: int = 20261031):
        """k_range=(5, 10): each row gets the reranked top-k with k drawn per row (the model sees varied context sizes).
        oracle=p: when the gold Điều is missing from that top-k, with probability p a gold passage (found in the dense
        top-`candidates`) replaces a random slot below the first (RAFT); the rest stay as retrieved (= the test case)."""
        self.cfg, self.name, self.splits = cfg, name, splits   # final model: splits=("train", "dev")
        self.k_range, self.oracle, self.rng = k_range, oracle, random.Random(seed)
        self.shots, self.refuse = shots, refuse
        self.out = OUT / name

    def build(self) -> Path:
        from transformers import AutoTokenizer

        self.stats = {}
        tok = AutoTokenizer.from_pretrained(self.cfg.generate.model)
        self.prompt = Prompt.load(self.cfg.generate.prompt)
        dc = Decontam()
        kept, report = dc.filter([q for s in self.splits for q in QASet().load(s)])
        self.out.mkdir(parents=True, exist_ok=True)
        self._write(tok, kept, self.out / "train.jsonl")
        if "dev" not in self.splits:
            val, vrep = dc.filter(QASet().load("dev"))
            self._write(tok, val, self.out / "val.jsonl", train=False)
            report["val"] = {k: vrep[k] for k in ("in", "kept", "dropped")}
        report |= {"splits": list(self.splits), "model": self.cfg.generate.model, "prompt": self.cfg.generate.prompt,
                   "retrieval_key": self.cfg.retrieval_key(), "k": self.cfg.retrieve.k, "k_range": self.k_range,
                   "oracle": self.oracle, "shots": self.shots, "refuse": self.refuse, "gold_in_context": self.stats}
        Decontam.save(report, self.out / "report.json")
        print(f"sft data: {len(kept)} train rows, {report.get('val', {}).get('kept', 0)} val rows -> {self.out}")
        return self.out

    def _write(self, tok, questions, path: Path, train: bool = True) -> None:
        kr, oracle = (self.k_range, self.oracle) if train else (None, 0.0)   # val = the test condition
        store = ShotStore() if (train and max(self.shots) > 0) else None
        top = Retrieval(self.cfg.model_copy(update={"retrieve": self.cfg.retrieve.model_copy(
            update={"k": kr[1] if kr else self.cfg.retrieve.k})})).run(questions)   # cached
        pool = (Retrieval(self.cfg.model_copy(update={
            "rerank": self.cfg.rerank.model_copy(update={"model": None}),
            "retrieve": self.cfg.retrieve.model_copy(update={"k": self.cfg.retrieve.candidates})})).run(questions)
                if oracle else {})
        ev, n, gold_in, injected, refused = RetrievalEval(), 0, 0, 0, 0
        qs = list(questions)

        def render(q, ctx, answer, avoid=frozenset()):
            shots = store.pick(q.text, self.rng.choice(self.shots), avoid) if store else []
            text = tok.apply_chat_template(self.prompt.messages(q.text, ctx, shots), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            return {"qid": q.qid, "prompt": text, "completion": answer.strip() + "<|im_end|>"}

        with open(path, "w", encoding="utf-8") as f:
            for q in qs:
                ctx = list(top[q.qid][: self.rng.randint(*kr)] if kr else top[q.qid])
                g = ev.gold.get(q.qid)
                has = lambda ps: bool(g and g.pairs) and any(a and f"{k}#{a}" in g.pairs for k, a in
                                                              (ev.passage_keys({"doc": p.doc, "text": p.text}) for p in ps))
                if g and g.pairs:
                    n += 1
                    if not has(ctx) and oracle and self.rng.random() < oracle:
                        gold = [p for p in pool.get(q.qid, []) if has([p])]
                        if gold:
                            ctx[self.rng.randint(1, len(ctx)) - 1 if len(ctx) > 1 else 0] = gold[0]
                            injected += 1
                    gold_in += has(ctx)
                avoid = frozenset(g.pairs) if g else frozenset()
                f.write(json.dumps(render(q, ctx, q.answer, avoid), ensure_ascii=False) + "\n")
                if train and self.refuse and self.rng.random() < self.refuse:
                    # no-answer case: the context was retrieved for an unrelated question (different law) -> refuse
                    other = self.rng.choice(qs)
                    og = ev.gold.get(other.qid)
                    if og and g and og.docs and g.docs and not (og.docs & g.docs):
                        octx = list(top[other.qid][: self.rng.randint(*kr)] if kr else top[other.qid])
                        f.write(json.dumps(render(q, octx, self.rng.choice(REFUSALS).format(q=q.text.strip()), avoid), ensure_ascii=False) + "\n")
                        refused += 1
        self.stats[path.stem] = {"with_gold": n, "gold_in_context": gold_in, "injected": injected, "refusal_rows": refused}
