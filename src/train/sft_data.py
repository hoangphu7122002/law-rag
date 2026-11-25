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
from train.ctx_mix import answerable_context, drop_gold, fit_budget, full_gold, place_gold_randomly, inject_all, maybe_shuffle, missing_gold

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
                 refuse: float = 0.0, seed: int = 20261031,
                 shuffle: float = 0.0, all_gold: bool = False, drop_gold_p: float = 0.0,
                 guarantee: bool = False, extra: str = "", teacher: str = "", max_tokens: int = 0, limit: int = 0, full_gold_articles: bool = False):
        """k_range=(5, 10): each row gets the reranked top-k with k drawn per row (the model sees varied context sizes).
        oracle=p: when the gold Điều is missing from that top-k, with probability p a gold passage (found in the dense
        top-`candidates`) replaces a random slot below the first (RAFT); the rest stay as retrieved (= the test case)."""
        self.cfg, self.name, self.splits = cfg, name, splits   # final model: splits=("train", "dev")
        self.k_range, self.oracle, self.rng = k_range, oracle, random.Random(seed)
        self.shots, self.refuse = shots, refuse
        self.shuffle, self.all_gold, self.drop_gold_p = shuffle, all_gold, drop_gold_p   # context mixing (train.ctx_mix)
        # guarantee: every non-refusal row has a gold article in its context (fetched from the corpus when retrieval missed it);
        # rows whose gold is unknown or not in the corpus are dropped, not trained on. extra: synthetic QA jsonl (train.synth_gen);
        # teacher: jsonl {"qid", "answer"} of verified good teacher answers for train questions (a second answer per question).
        self.guarantee, self.extra, self.teacher = guarantee, extra, teacher
        self.full_gold = full_gold_articles   # with guarantee: every gold article in the context in FULL (whole article, all of them)
        self.limit = limit   # smoke test: only the first N train questions and the first N synthetic rows
        self.max_tokens = max_tokens   # rows longer than this (prompt + answer) lose non-gold passages from the tail; the trainer's max_length must be >= this
        if guarantee and drop_gold_p:
            raise ValueError("guarantee and drop_gold are contradictory")
        self.out = OUT / name

    def build(self) -> Path:
        from transformers import AutoTokenizer

        self.stats = {}
        tok = AutoTokenizer.from_pretrained(self.cfg.generate.model)
        self.prompt = Prompt.load(self.cfg.generate.prompt)
        dc = Decontam()
        kept, report = dc.filter([q for s in self.splits for q in QASet().load(s)])
        self.extra_gold = {}
        if self.limit:
            kept = kept[: self.limit]
        extras = self._extra_questions()
        kept = kept + (extras[: self.limit] if self.limit else extras) + self._teacher_questions(kept)
        self.out.mkdir(parents=True, exist_ok=True)
        self._write(tok, kept, self.out / "train.jsonl")
        if "dev" not in self.splits:
            val, vrep = dc.filter(QASet().load("dev"))
            self._write(tok, val, self.out / "val.jsonl", train=False)
            report["val"] = {k: vrep[k] for k in ("in", "kept", "dropped")}
        report |= {"splits": list(self.splits), "model": self.cfg.generate.model, "prompt": self.cfg.generate.prompt,
                   "retrieval_key": self.cfg.retrieval_key(), "k": self.cfg.retrieve.k, "k_range": self.k_range,
                   "guarantee": self.guarantee, "full_gold": self.full_gold, "max_tokens": self.max_tokens, "extra": self.extra, "teacher": self.teacher, "oracle": self.oracle, "shots": self.shots, "refuse": self.refuse, "shuffle": self.shuffle, "all_gold": self.all_gold, "drop_gold": self.drop_gold_p, "gold_in_context": self.stats}
        Decontam.save(report, self.out / "report.json")
        print(f"sft data: {len(kept)} train rows, {report.get('val', {}).get('kept', 0)} val rows -> {self.out}")
        return self.out

    def _write(self, tok, questions, path: Path, train: bool = True) -> None:
        kr, oracle = (self.k_range, self.oracle) if train else (None, 0.0)   # val = the test condition
        store = ShotStore() if (train and max(self.shots) > 0) else None
        top = Retrieval(self.cfg.model_copy(update={"retrieve": self.cfg.retrieve.model_copy(
            update={"k": kr[1] if kr else self.cfg.retrieve.k})})).run(list({q.qid: q for q in questions}.values()))   # cached
        pool = (Retrieval(self.cfg.model_copy(update={
            "rerank": self.cfg.rerank.model_copy(update={"model": None}),
            "retrieve": self.cfg.retrieve.model_copy(update={"k": self.cfg.retrieve.candidates})})).run(list({q.qid: q for q in questions}.values()))
                if oracle else {})
        ev, n, gold_in, injected, refused = RetrievalEval(), 0, 0, 0, 0
        ev.gold = {**ev.gold, **self.extra_gold}
        qs = list(questions)
        skipped = {"unanswerable": 0, "unverifiable": 0, "too_long": 0}
        guard = self.guarantee and train
        lookup = self._article_lookup({k for q in qs for k in (ev.gold[q.qid].pairs if q.qid in ev.gold else ())}) if guard else None

        key_of = lambda p: "%s#%s" % ev.passage_keys({"doc": p.doc, "text": p.text})   # noqa: E731
        size = lambda p: len(tok(p.text, add_special_tokens=False)["input_ids"]) + 12   # noqa: E731

        def fitted(q, ctx, answer, avoid, is_gold):
            """render + shrink to max_tokens (None: the gold alone does not fit)."""
            row = render(q, ctx, answer, avoid)
            for _ in range(3):
                n_tok = len(tok(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"])
                if not (train and self.max_tokens) or n_tok <= self.max_tokens:
                    return row
                ctx = fit_budget(ctx, is_gold, size, sum(size(p) for p in ctx) - (n_tok - self.max_tokens))
                if ctx is None:
                    return None
                row = render(q, ctx, answer, avoid)
            return None

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
                        if gold and self.all_gold:   # every gold article the context lacks (multi-article references), not just one
                            ctx = inject_all(ctx, missing_gold(ctx, pool.get(q.qid, []), key_of, g.pairs), self.rng)
                            injected += 1
                        elif gold:
                            ctx[self.rng.randint(1, len(ctx)) - 1 if len(ctx) > 1 else 0] = gold[0]
                            injected += 1
                    elif train and self.drop_gold_p and has(ctx) and self.rng.random() < self.drop_gold_p:   # a retrieval miss, reference kept
                        spares = [p for p in top[q.qid] if p not in ctx and not has([p])]
                        ctx = drop_gold(ctx, lambda p: has([p]), spares, self.rng)
                    gold_in += has(ctx)
                if guard:
                    fixed = answerable_context(ctx, set(g.pairs) if g else set(), key_of, has, lookup, self.rng)
                    if fixed is None:
                        skipped["unverifiable" if not (g and g.pairs) else "unanswerable"] += 1
                        continue
                    gold_in += not has(ctx)   # gold put in from the corpus (the earlier count covers rows that already had it)
                    ctx = full_gold(fixed, set(g.pairs), key_of, lookup, self.rng) if self.full_gold else fixed
                    if self.full_gold:   # the gold articles at random positions, not where retrieval left them (mostly the top)
                        ctx = place_gold_randomly(ctx, lambda p: key_of(p) in g.pairs, self.rng)
                if train:
                    ctx = maybe_shuffle(ctx, self.shuffle, self.rng)
                avoid = frozenset(g.pairs) if g else frozenset()
                row = fitted(q, ctx, q.answer, avoid, lambda p: has([p]))
                if row is None:
                    skipped["too_long"] += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if train and self.refuse and self.rng.random() < self.refuse:
                    # no-answer case: the context was retrieved for an unrelated question (different law) -> refuse
                    other = self.rng.choice(qs)
                    og = ev.gold.get(other.qid)
                    if og and g and og.docs and g.docs and not (og.docs & g.docs):
                        octx = list(top[other.qid][: self.rng.randint(*kr)] if kr else top[other.qid])
                        rrow = fitted(q, octx, self.rng.choice(REFUSALS).format(q=q.text.strip()), avoid, lambda p: False)
                        if rrow is not None:
                            f.write(json.dumps(rrow, ensure_ascii=False) + "\n")
                            refused += 1
        self.stats[path.stem] = {"with_gold": n, "gold_in_context": gold_in, "injected": injected, "refusal_rows": refused, "skipped": skipped}

    def _extra_questions(self) -> list:
        """Synthetic QA (train.synth_gen output) as training questions; their gold = the articles they were written from."""
        if not self.extra:
            return []
        from corpus.gold import Gold
        from records import Question

        out = []
        for i, line in enumerate(Path(self.extra).read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            if not r.get("ok", True) or not r.get("sources"):
                continue
            qid = 10_000_000 + i
            pairs = frozenset(r["sources"])
            self.extra_gold[qid] = Gold(qid, frozenset(k.split("#")[0] for k in pairs), pairs, sorted(pairs)[0].split("#")[0])
            out.append(Question(qid, r["question"], r["answer"], ""))
        return out

    def _teacher_questions(self, kept: list) -> list:
        """A second answer for train questions: the same question with a verified good teacher answer (qid shared, so gold/retrieval are reused)."""
        if not self.teacher:
            return []
        from records import Question

        by_qid = {q.qid: q for q in kept}
        out = []
        for line in Path(self.teacher).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            q = by_qid.get(int(r["qid"]))
            if q is not None and r.get("answer"):
                out.append(Question(q.qid, q.text, r["answer"], q.law_id))
        return out

    def _article_lookup(self, keys: set):
        """key "<law>#<Điều>" -> one Passage with the whole article (all its chunks) straight from the corpus nodes."""
        from llama_index.core.schema import MetadataMode

        from index.store import IndexStore
        from records import Passage

        parts: dict[str, list] = {}
        for n in IndexStore(self.cfg).nodes():
            law, art = RetrievalEval.passage_keys({"doc": n.metadata.get("doc", ""), "text": n.get_content()[:400]})
            k = f"{law}#{art}" if art else None
            if k in keys:
                parts.setdefault(k, []).append(n)
        out = {}
        for k, ns in parts.items():
            text = "\n".join([ns[0].get_content(metadata_mode=MetadataMode.LLM)] + [n.get_content() for n in ns[1:]])
            out[k] = Passage(pid=ns[0].node_id, doc=ns[0].metadata.get("doc", ""), text=text)
        return out.get
