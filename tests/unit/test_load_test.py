import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from load_test import parse_sse_line, summarize_load


def test_summary_throughput_percentiles_and_errors():
    recs = [{"ttft": 0.2, "total": 1.0}, {"ttft": 0.4, "total": 2.0}, {"ttft": 0.6, "total": 3.0}, {"ttft": 1.0, "total": 10.0},
            {"status": 429}, {"status": 503}, {"status": 404}]
    s = summarize_load(recs, wall=4.0, users=8)
    assert s["users"] == 8 and s["ok"] == 4 and s["rejected_429"] == 1 and s["errors"] == 2
    assert s["rps"] == 1.0                                  # 4 completed / 4 s
    assert s["ttft_p50"] == 0.6 and s["total_p95"] == 10.0 and s["total_p50"] == 3.0


def test_empty_run_does_not_divide_by_zero():
    s = summarize_load([], wall=1.0, users=1)
    assert s["ok"] == 0 and s["rps"] == 0.0 and s["total_p50"] is None


def test_sse_lines_are_parsed_into_event_and_json():
    assert parse_sse_line("event: token") == ("event", "token")
    assert parse_sse_line('data: {"text": "a"}') == ("data", {"text": "a"})
    assert parse_sse_line("") is None
