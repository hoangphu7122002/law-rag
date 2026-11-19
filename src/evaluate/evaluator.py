"""Run directory -> per-question scores + means. Reads predictions.jsonl, so scoring never re-runs the LLM."""
import time

from corpus import QASet
from corpus.keys import law_key, stem_key
from evaluate import metrics
from runs import Ledger, Run

COSINE_MODEL = "keepitreal/vietnamese-sbert"  # the course notebook's choice; keep it for a fair comparison
METRICS = ["cosine", "jaccard", "token_overlap", "bleu1", "rouge_l"]


class Evaluator:
    def __init__(self, cosine: bool = True, device: str = "cpu"):
        self.cosine, self.device = cosine, device
        self._encoder = None

    def score(self, run: Run) -> dict:
        cfg = run.config()
        refs = {q.qid: q for q in QASet().load(cfg["split"], cfg["limit"])}
        rows = [self.score_one(p, refs[p["qid"]]) for p in run.predictions() if p["qid"] in refs]
        means = {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in METRICS + ["law_hit"]}
        summary = {"n": len(rows), "errors": sum(bool(r["error"]) for r in rows), **means}
        run.write_json("metrics.json", {"summary": summary, "rows": rows})
        Ledger().append({"time": time.strftime("%Y-%m-%d %H:%M"), "run": run.path.name, "split": cfg["split"],
                         "git": run.read_json("meta.json")["git"], "note": cfg.get("note", ""), **summary})
        return summary

    def score_one(self, pred: dict, ref) -> dict:
        answer = pred["answer"]
        row = {"qid": pred["qid"], "error": pred.get("error", ""), **metrics.all_lexical(answer, ref.answer)}
        row["cosine"] = metrics.cosine_similarity_score(answer, ref.answer, self.embed) if self.cosine else 0.0
        row["law_hit"] = float(self.law_hit(ref.law_id, pred.get("passages", [])))
        return row

    @staticmethod
    def law_hit(law_id: str, docs: list[str]) -> bool:
        """Was the cited document among the passages given to the LLM?"""
        target = law_key(law_id)
        return bool(target) and any(stem_key(d) == target for d in docs)

    def embed(self, text: str):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(COSINE_MODEL, device=self.device)
        return self._encoder.encode(text)
