import numpy as np
import pandas as pd

from benchmark_suite.metrics import (
    hypervolume_2d,
    hypervolume_ratio,
    epsilon_recall,
    pareto_recall,
    error_agreement,
)


def _front(points):
    return pd.DataFrame({
        "poly_degree": 8192,
        "scale": 2**30,
        "chain": [(i,) for i in range(len(points))],
        "runtime": [p[0] for p in points],
        "max_err": [p[1] for p in points],
    })


def test_hypervolume_2d():
    pts = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert hypervolume_2d(pts, np.array([2.0, 2.0])) == 3.0


def test_identical_fronts_have_ratio_one():
    front = _front([(1.0, 1e-2), (2.0, 1e-6)])
    assert hypervolume_ratio(front, front) == 1.0
    assert pareto_recall(front, front) == 1.0
    assert epsilon_recall(front, front) == 1.0


def test_ratio_is_sensitive_to_missing_points():
    full = _front([(1.0, 1e-2), (2.0, 1e-6)])
    tool = _front([(1.0, 1e-2)])
    assert hypervolume_ratio(full, tool) < 0.6


def test_error_agreement_perfect():
    est = pd.Series([1e-1, 1e-3, 1e-5, 1e-7])
    result = error_agreement(est, est)
    assert result["spearman"] == 1.0
    assert result["r2_log"] == 1.0
    assert result["coverage"] == 1.0


def test_adding_tool_points_never_lowers_ratio():
    full = _front([(1.0, 1e-2), (1.5, 1e-4), (2.0, 1e-6)])
    tool_small = _front([(1.2, 1e-1)])
    tool_more = _front([(1.2, 1e-1), (1.5, 1e-4)])
    assert hypervolume_ratio(full, tool_more) >= hypervolume_ratio(full, tool_small)
