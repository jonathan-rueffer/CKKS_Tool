import pandas as pd

from ckks_param_tool.search.pareto import pareto_mask, select_representative_pareto


def test_pareto_mask():
    df = pd.DataFrame({"runtime": [1, 2, 3, 2], "max_err": [3, 2, 1, 3]})
    assert pareto_mask(df, ["runtime", "max_err"]).tolist() == [True, True, True, False]


def test_duplicates_are_both_kept():
    df = pd.DataFrame({"runtime": [1, 1], "max_err": [1, 1]})
    assert pareto_mask(df, ["runtime", "max_err"]).all()


def test_knee_uses_log_error():
    # on a linear error scale the middle point would look far from the ideal point
    df = pd.DataFrame({"runtime": [1.0, 1.1, 3.0], "max_err": [1e-1, 1e-5, 1e-6]})
    assert select_representative_pareto(df).name == 1


def test_knee_treats_runtimes_within_tolerance_as_equal():
    # 0.6% slower but 10 bits more precise: within noise, so the precise point is recommended
    df = pd.DataFrame({"runtime": [3.284, 3.303], "max_err": [2**-22.9, 2**-33.4]})
    assert select_representative_pareto(df).name == 1
    assert select_representative_pareto(df, runtime_tol=0.0).name == 0


def test_two_point_tie_goes_to_faster_point_regardless_of_order():
    df = pd.DataFrame({"runtime": [0.0187, 0.0051], "max_err": [7.7e-10, 3.6e-9]})
    assert select_representative_pareto(df).name == 1
    assert select_representative_pareto(df.iloc[::-1]).name == 1
