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


def test_endpoints_respect_tolerances():
    from ckks_param_tool.search.pareto import fastest_point, most_precise_point, fastest_meeting

    df = pd.DataFrame({"runtime": [1.00, 1.01, 2.0, 4.0], "max_err": [2.0**-10, 2.0**-20, 2.0**-30, 2.0**-31]})
    # 1% slower counts as equally fast at a 2% tolerance: the more precise of the two
    assert fastest_point(df, runtime_tol=0.02).name == 1
    assert fastest_point(df, runtime_tol=0.0).name == 0
    # 1 bit less precise counts as equally precise at 1.5 bits: the faster of the two
    assert most_precise_point(df, precision_tol_bits=1.5).name == 2
    assert most_precise_point(df, precision_tol_bits=0.0).name == 3
    assert fastest_meeting(df, 25, runtime_tol=0.0).name == 2
    assert fastest_meeting(df, 40) is None


def test_knee_does_not_depend_on_row_order():
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"runtime": np.sort(rng.uniform(1, 5, 8)), "max_err": np.sort(2.0 ** -rng.uniform(5, 40, 8))[::-1]})
    knee = select_representative_pareto(df, runtime_tol=0.01).name
    for seed in range(5):
        shuffled = df.sample(frac=1, random_state=seed)
        assert select_representative_pareto(shuffled, runtime_tol=0.01).name == knee
