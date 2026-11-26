import sys
from pathlib import Path

from cascade_stats import cascade_curve


def test_cascade_curve_endpoints_and_cost():
    rows = [{"signal": s, "greedy": g, "mbr": m} for s, g, m in [(0.1, 0.4, 0.8), (0.2, 0.5, 0.7), (0.9, 0.7, 0.7), (0.8, 0.6, 0.6)]]
    c = cascade_curve(rows, fractions=(0.0, 0.5, 1.0))
    assert c[0.0]["rouge"] == 0.55 and c[0.0]["cost"] == 1.0          # greedy for everyone
    assert c[1.0]["rouge"] == 0.7 and c[1.0]["cost"] == 8.0           # MBR8 for everyone
    assert c[0.5]["rouge"] == 0.7 and c[0.5]["cost"] == 4.5           # MBR only for the 2 lowest signals (low margin = unsure)
