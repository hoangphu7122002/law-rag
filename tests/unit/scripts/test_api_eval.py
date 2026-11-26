import sys
from pathlib import Path

from api_eval import summarize


def test_summary_has_mean_rouge_and_latency_percentiles():
    rows = [{"rouge": 0.5, "seconds": 1.0}, {"rouge": 0.7, "seconds": 2.0}, {"rouge": 0.6, "seconds": 3.0}, {"rouge": 0.8, "seconds": 10.0}]
    s = summarize(rows)
    assert s["n"] == 4 and s["rouge_l"] == 0.65
    assert s["p50_s"] == 3.0 and s["p95_s"] == 10.0 and s["mean_s"] == 4.0


def test_failed_requests_are_counted_not_averaged():
    s = summarize([{"rouge": 0.6, "seconds": 1.0}, {"error": "503", "seconds": 0.1}])
    assert s["n"] == 1 and s["errors"] == 1
