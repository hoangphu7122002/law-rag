from serve.metrics import Metrics


def test_counters_gauge_and_histogram_render_in_prometheus_text():
    m = Metrics(buckets=(0.5, 1.0, 2.0))
    m.inc("requests_total", route="/ask", status="200")
    m.inc("requests_total", route="/ask", status="200")
    m.gauge("in_flight", 3)
    for v in (0.3, 0.7, 5.0):
        m.observe("latency_seconds", v, route="/ask")
    text = m.render()
    assert 'lawrag_requests_total{route="/ask",status="200"} 2' in text
    assert "lawrag_in_flight 3" in text
    assert 'lawrag_latency_seconds_bucket{route="/ask",le="0.5"} 1' in text
    assert 'lawrag_latency_seconds_bucket{route="/ask",le="1.0"} 2' in text
    assert 'lawrag_latency_seconds_bucket{route="/ask",le="+Inf"} 3' in text
    assert 'lawrag_latency_seconds_count{route="/ask"} 3' in text
    assert 'lawrag_latency_seconds_sum{route="/ask"} 6' in text
