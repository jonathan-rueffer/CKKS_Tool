import pytest
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


def test_runtime_floor_ignores_noise_on_flat_fronts():
    # the whole front lies within 2% runtime: missing the 1%-faster point should cost little
    full = _front([(2.067, 2**-2), (2.087, 2**-30), (2.110, 2**-33)])
    tool = _front([(2.087, 2**-30), (2.110, 2**-33)])
    assert hypervolume_ratio(full, tool) > 0.9
    assert hypervolume_ratio(full, tool, runtime_tol=0.0) < hypervolume_ratio(full, tool)


def test_dominating_configs():
    from benchmark_suite.metrics import dominating_configs

    cands = _front([(1.0, 1e-3), (2.0, 1e-6), (4.0, 1e-9)])
    fastest, most_precise = dominating_configs(cands, baseline_err=1e-5, baseline_runtime=3.0)
    assert fastest["runtime"] == 2.0
    assert most_precise["max_err"] == 1e-6


def test_epsilon_indicator():
    from benchmark_suite.metrics import epsilon_indicator

    ref = _front([(1.0, 1e-6), (2.0, 1e-9)])
    assert epsilon_indicator(ref, ref) == 1.0
    # the tool only has a point 10% slower and twice the error of the first reference point
    tool = _front([(1.1, 2e-6)])
    # the second reference point (2.0, 1e-9) needs factor max(1.1/2, 2e-6/1e-9) = 2000
    assert epsilon_indicator(ref, tool) == pytest.approx(2000.0)
    assert epsilon_indicator(ref, _front([(1.1, 2e-6), (2.0, 1e-9)])) == pytest.approx(2.0)


def test_hypervolume_ratio_by_hand():
    # reference front (1, 1e-2), (3, 1e-6) in (runtime, error): normalized by its ideal (1, -6)
    # and span (2, 4) in (runtime, log10 error) to (0, 1) and (1, 0); reference point (1.1, 1.1).
    # HV(reference) = 1.1 * 0.1 + 0.1 * 1.0 = 0.21
    ref = _front([(1.0, 1e-2), (3.0, 1e-6)])
    assert hypervolume_2d(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1.1, 1.1])) == pytest.approx(0.21)
    # the tool point (2, 1e-4) normalizes to (0.5, 0.5): HV = 0.6 * 0.6. A ratio above 1 is possible
    # because the tool point is not dominated by the reference front's staircase
    assert hypervolume_ratio(ref, _front([(2.0, 1e-4)])) == pytest.approx(0.36 / 0.21)


def test_hypervolume_points_beyond_the_reference_point_count_nothing():
    ref = _front([(1.0, 1e-2), (3.0, 1e-6)])
    slow = _front([(10.0, 1e-9)]) # normalized runtime 4.5 > 1.1
    assert hypervolume_ratio(ref, slow) == 0.0
    assert hypervolume_ratio(ref, _front([])) == 0.0


def test_hypervolume_floor_uses_the_given_tolerance():
    # a front 1% wide in runtime: with a 5% floor the runtime axis is stretched less than with 1%
    ref = _front([(1.00, 1e-2), (1.01, 1e-6)])
    tool = _front([(1.01, 1e-6)])
    r5 = hypervolume_ratio(ref, tool, runtime_tol=0.05)
    r1 = hypervolume_ratio(ref, tool, runtime_tol=0.01)
    r0 = hypervolume_ratio(ref, tool, runtime_tol=0.0)
    assert r5 > r1 >= r0


def test_epsilon_recall_tolerance_is_inclusive():
    ref = _front([(1.0, 2.0**-20)])
    rt_tol, bits_tol = 0.04, 2.0
    on_edge = _front([(1.04, 2.0**-18)])         # exactly (1 + rt_tol) slower, bits_tol less precise
    beyond_rt = _front([(1.0401, 2.0**-20)])
    beyond_bits = _front([(1.0, 2.0**-17.99)])
    assert epsilon_recall(ref, on_edge, rt_tol, bits_tol) == 1.0
    assert epsilon_recall(ref, beyond_rt, rt_tol, bits_tol) == 0.0
    assert epsilon_recall(ref, beyond_bits, rt_tol, bits_tol) == 0.0
    assert epsilon_recall(ref, _front([]), rt_tol, bits_tol) == 0.0


def test_epsilon_recall_is_a_fraction_of_reference_points():
    ref = _front([(1.0, 1e-2), (2.0, 1e-6), (3.0, 1e-9)])
    tool = _front([(1.0, 1e-2), (2.0, 1e-6)])
    assert epsilon_recall(ref, tool, 0.0, 0.0) == pytest.approx(2 / 3)


def test_epsilon_indicator_direction():
    from benchmark_suite.metrics import epsilon_indicator

    ref = _front([(2.0, 1e-6)])
    # a tool front that dominates the reference gives eps < 1, a worse one eps > 1
    assert epsilon_indicator(ref, _front([(1.0, 1e-7)])) == pytest.approx(0.5)
    assert epsilon_indicator(ref, _front([(4.0, 1e-6)])) == pytest.approx(2.0)
    # the indicator is the worst case over reference points of the best tool point
    two = _front([(2.0, 1e-6), (1.0, 1e-3)])
    assert epsilon_indicator(two, _front([(2.0, 1e-6)])) == pytest.approx(2.0)


def test_pareto_recall_is_exact_membership():
    ref = _front([(1.0, 1e-2), (2.0, 1e-6)])
    same_keys_other_values = ref.assign(runtime=[5.0, 5.0])
    assert pareto_recall(ref, same_keys_other_values) == 1.0 # keys, not measurements
    assert pareto_recall(ref, ref.iloc[:1]) == 0.5


def test_paired_front_uses_the_grid_measurements_of_the_tool_candidates():
    from benchmark_suite.metrics import keys, restrict_to, pareto_front, at_least

    grid = _front([(1.0, 1e-2), (2.0, 1e-6), (3.0, 1e-9)]).assign(security=[128, 192, 256])
    tool = grid.iloc[[0, 2]].assign(runtime=[9.0, 9.0]) # the tool's own (different) timings
    paired = restrict_to(grid, keys(tool))
    assert paired["runtime"].tolist() == [1.0, 3.0] # grid measurements, not the tool's
    # "at least" a security level: a 256-bit configuration also competes at 128 and 192
    assert len(at_least(paired, 192)) == 1 and len(at_least(paired, 128)) == 2
    assert len(pareto_front(paired)) == 2


def test_grid_position_tolerances():
    from benchmark_suite.analyze import _grid_position

    def bits(df):
        return df.assign(precision_bits=-np.log2(df["max_err"]))

    grid = bits(_front([(1.0, 2.0**-20), (1.5, 2.0**-30), (3.0, 2.0**-40)]).assign(security=128))
    key = lambda i: tuple(grid.iloc[i][["poly_degree", "scale", "chain"]])
    assert _grid_position(grid, key(1), 128, 0.04, 2.0) == (True, True)
    # a point that is beaten by more than both tolerances is neither on nor near the front
    worse = bits(pd.concat([grid, _front([(1.2, 2.0**-10)]).assign(chain=[(9,)], security=128)], ignore_index=True))
    key_w = tuple(worse.iloc[3][["poly_degree", "scale", "chain"]])
    assert _grid_position(worse, key_w, 128, 0.04, 2.0) == (False, False)
    # beaten only within the tolerances: not on the front, but near it
    close = bits(pd.concat([grid, _front([(1.03, 2.0**-19)]).assign(chain=[(8,)], security=128)], ignore_index=True))
    key_c = tuple(close.iloc[3][["poly_degree", "scale", "chain"]])
    assert _grid_position(close, key_c, 128, 0.04, 2.0) == (False, True)
    assert _grid_position(None, key(0), 128, 0.04, 2.0) == (None, None)


def test_analysis_tables_are_deterministic(tmp_path):
    # the same raw measurements give bit-identical summaries
    from benchmark_suite.analyze import search_efficiency
    run = _fake_run(tmp_path)
    a = pd.DataFrame(search_efficiency("fake", run))
    b = pd.DataFrame(search_efficiency("fake", run))
    pd.testing.assert_frame_equal(a, b)
    assert (a["tool_keys_missing_from_broad_grid"] == 0).all()


def _fake_run(tmp_path):
    import json
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        rows.append({"poly_degree": 8192, "scale": 2**(20 + i % 9 * 5), "log_scale": 20 + i % 9 * 5,
                     "chain": str((60, 20 + i % 9 * 5, 60 + i // 9)), "max_err": float(2.0 ** -rng.uniform(5, 40)),
                     "precision_bits": 0.0, "runtime": float(rng.uniform(1, 3)), "security": [128, 192, 256][i % 3],
                     "eval_seconds": 1.0, "status": "success"})
    grid = pd.DataFrame(rows)
    run = tmp_path / "run"
    for sub, df in (("broad_grid", grid), ("tool", grid.iloc[::4])):
        (run / sub).mkdir(parents=True)
        df.to_csv(run / sub / "param_search.csv", index=False)
        (run / sub / "run_info.json").write_text(json.dumps({"search_seconds": float(len(df))}))
        (run / sub / "config.json").write_text(json.dumps({"evaluation": {"runtime_tolerance": 0.01,
                                                                         "precision_tolerance_bits": 2.5}}))
    return run
