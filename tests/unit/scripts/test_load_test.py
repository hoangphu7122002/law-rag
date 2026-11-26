import sys
from pathlib import Path

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


def test_gpu_samples_are_parsed_and_averaged_per_gpu():
    from load_test import mean_by_gpu, parse_smi

    a = parse_smi("0, 50\n1, 4\n")
    b = parse_smi("0, 70\n1, 0\n")
    assert a == {0: 50.0, 1: 4.0}
    assert mean_by_gpu([a, b]) == {"gpu0_util": 60.0, "gpu1_util": 2.0}
    assert mean_by_gpu([]) == {}


def test_a_level_with_no_success_is_flagged_as_broken():
    from load_test import level_problem

    assert "no request succeeded" in level_problem({"ok": 0, "errors": 5, "rejected_429": 0, "gpu0_util": 0.0})
    assert level_problem({"ok": 10, "errors": 0, "rejected_429": 0, "gpu0_util": 55.0}) is None
    assert "idle" in level_problem({"ok": 10, "errors": 0, "rejected_429": 0, "gpu0_util": 1.0, "gpu1_util": 0.0})


def test_find_max_users_doubles_then_bisects_to_the_slo_boundary():
    from load_test import find_max_users

    calls = []

    def measure(u):                      # SLO holds up to 37 users
        calls.append(u)
        return {"users": u, "ok": 10, "errors": 0, "rejected_429": 0, "total_p95": 5.0 if u <= 37 else 20.0}

    best, rows = find_max_users(measure, start=2, limit=256, slo_p95=10.0)
    assert best == 37
    assert calls[:5] == [2, 4, 8, 16, 32] and 64 in calls and len(calls) < 15
    assert [r["users"] for r in rows] == calls


def test_find_max_users_respects_limit_and_reports_zero_when_even_start_fails():
    from load_test import find_max_users

    ok = lambda u: {"users": u, "ok": 5, "errors": 0, "rejected_429": 0, "total_p95": 1.0}  # noqa: E731
    assert find_max_users(ok, start=4, limit=32, slo_p95=10.0)[0] == 32
    bad = lambda u: {"users": u, "ok": 5, "errors": 0, "rejected_429": 0, "total_p95": 99.0}  # noqa: E731
    assert find_max_users(bad, start=4, limit=32, slo_p95=10.0)[0] == 0


def test_slo_fails_on_errors_or_rejections():
    from load_test import meets_slo

    assert meets_slo({"ok": 99, "errors": 0, "rejected_429": 0, "total_p95": 3.0}, 10.0)
    assert not meets_slo({"ok": 90, "errors": 10, "rejected_429": 0, "total_p95": 3.0}, 10.0)
    assert not meets_slo({"ok": 99, "errors": 0, "rejected_429": 5, "total_p95": 3.0}, 10.0)
    assert not meets_slo({"ok": 0, "errors": 3, "rejected_429": 0, "total_p95": None}, 10.0)


def test_phases_parse_and_arrivals_follow_each_rate():
    from load_test import parse_phases, poisson_arrivals

    ph = parse_phases("0.5:20,4:10,0.5:20")
    assert ph == [(0.5, 20.0), (4.0, 10.0), (0.5, 20.0)]
    arr = poisson_arrivals(ph, seed=1)
    assert arr == sorted(arr) and all(0 <= t < 50 for t, _ in arr)
    counts = [sum(1 for _, p in arr if p == i) for i in range(3)]
    assert 4 <= counts[0] <= 18 and 25 <= counts[1] <= 55 and 4 <= counts[2] <= 18   # ~10 / ~40 / ~10 expected
    assert [p for _, p in arr] == sorted(p for _, p in arr)                          # phases appear in order


def test_arrivals_are_reproducible():
    from load_test import parse_phases, poisson_arrivals

    ph = parse_phases("2:5")
    assert poisson_arrivals(ph, seed=3) == poisson_arrivals(ph, seed=3)


def test_phase_summary_reports_each_phase_and_recovery_ratio():
    from load_test import summarize_phases

    recs = ([{"phase": 0, "ttft": 0.3, "total": 3.0}] * 5 + [{"phase": 1, "ttft": 2.0, "total": 12.0}] * 5 + [{"phase": 1, "status": 429}] * 2
            + [{"phase": 2, "ttft": 0.4, "total": 4.5}] * 5)
    out = summarize_phases(recs, [(0.5, 10), (3.0, 10), (0.5, 10)])
    assert [p["phase"] for p in out["phases"]] == [0, 1, 2]
    assert out["phases"][1]["rejected_429"] == 2 and out["phases"][1]["total_p95"] == 12.0
    assert out["recovery_p95_ratio"] == 1.5                 # post-burst p95 / pre-burst p95 = 4.5 / 3.0


def test_the_most_common_client_error_is_reported():
    s = summarize_load([{"status": -1, "error": "ConnectError"}, {"status": -1, "error": "ConnectError"}, {"status": -1, "error": "ReadTimeout"}], 1.0, 8)
    assert s["top_error"] == "ConnectError"
    assert summarize_load([{"ttft": 1, "total": 2}], 1.0, 1)["top_error"] is None


def test_temperature_is_tracked_per_gpu_and_a_hot_gpu_stops_the_run():
    from load_test import level_problem, max_by_gpu, parse_smi_temp

    assert parse_smi_temp("0, 58\n1, 71\n") == {0: 58.0, 1: 71.0}
    assert max_by_gpu([{0: 58.0, 1: 71.0}, {0: 60.0, 1: 69.0}], "temp") == {"gpu0_temp": 60.0, "gpu1_temp": 71.0}
    hot = {"ok": 10, "errors": 0, "rejected_429": 0, "gpu0_util": 90.0, "gpu0_temp": 86.0, "gpu1_temp": 60.0}
    assert "too hot" in level_problem(hot)
    assert level_problem(hot | {"gpu0_temp": 70.0}) is None


def test_p99_is_reported_next_to_p95():
    recs = [{"ttft": i / 100, "total": float(i)} for i in range(1, 101)]
    s = summarize_load(recs, wall=10.0, users=4)
    assert s["total_p95"] == 96.0 and s["total_p99"] == 100.0
    assert s["ttft_p99"] == 1.0
