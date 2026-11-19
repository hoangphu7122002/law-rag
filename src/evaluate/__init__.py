"""Score a run, compare two runs.

metrics          the course notebook metrics, ported line for line (never change them)
Evaluator        run dir -> metrics.json (+ law hit@k) and one row in data/runs/results.tsv
RetrievalEval    retrieval only: doc@k / Điều@k vs the gold pairs parsed from the answers (corpus.gold)
PairedBootstrap  is run B better than run A on the same questions, beyond noise?
"""
from evaluate.bootstrap import PairedBootstrap
from evaluate.evaluator import Evaluator
from evaluate.retrieval_eval import RetrievalEval

__all__ = ["Evaluator", "PairedBootstrap", "RetrievalEval"]
