"""Tiny Prometheus text exposition (counters, gauges, histograms): no extra dependency, thread-safe enough for one event loop."""
from collections import defaultdict

DEFAULT_BUCKETS = (0.25, 0.5, 1, 2, 4, 8, 16, 32)


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in d.items()) + "}" if d else ""


class Metrics:
    def __init__(self, buckets=DEFAULT_BUCKETS, prefix: str = "lawrag_"):
        self.buckets, self.prefix = tuple(buckets), prefix
        self.counters: dict[tuple, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.hist: dict[tuple, list[float]] = {}

    def inc(self, name: str, n: float = 1, **labels) -> None:
        self.counters[(name, tuple(labels.items()))] += n

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value: float, **labels) -> None:
        h = self.hist.setdefault((name, tuple(labels.items())), [0.0] * (len(self.buckets) + 2))  # buckets..., count, sum
        for i, b in enumerate(self.buckets):
            if value <= b:
                h[i] += 1
        h[-2] += 1
        h[-1] += value

    def render(self) -> str:
        p, out = self.prefix, []
        for (name, lab), v in sorted(self.counters.items()):
            out.append(f"{p}{name}{_labels(dict(lab))} {v:g}")
        for name, v in sorted(self.gauges.items()):
            out.append(f"{p}{name} {v:g}")
        for (name, lab), h in sorted(self.hist.items()):
            lab = dict(lab)
            for b, c in zip(self.buckets, h, strict=False):
                out.append(f"{p}{name}_bucket{_labels({**lab, 'le': str(float(b))})} {c:g}")
            out.append(f"{p}{name}_bucket{_labels({**lab, 'le': '+Inf'})} {h[-2]:g}")
            out.append(f"{p}{name}_count{_labels(lab)} {h[-2]:g}")
            out.append(f"{p}{name}_sum{_labels(lab)} {h[-1]:g}")
        return "\n".join(out) + "\n"
