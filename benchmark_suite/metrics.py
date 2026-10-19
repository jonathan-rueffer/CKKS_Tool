import ast
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ckks_param_tool.search.pareto import pareto_mask

'''
Metrics used to evaluate the tool against the near-exhaustive search and to validate the
representative test vectors.

Objectives are (runtime, log10(max_err)), both minimized. Error is compared on a log scale
because it spans many orders of magnitude across scales.
'''

KEY = ["poly_degree", "scale", "chain"]


def load_param_search(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["chain"] = df["chain"].apply(lambda c: tuple(ast.literal_eval(c)) if isinstance(c, str) else tuple(c))
    for col in ("max_err", "runtime", "security"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def successful(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["status"] == "success") & np.isfinite(df["max_err"]) & np.isfinite(df["runtime"])]
    return df.copy()


def keys(df: pd.DataFrame) -> set:
    return set(zip(*(df[c] for c in KEY)))


def restrict_to(df: pd.DataFrame, key_set: set) -> pd.DataFrame:
    mask = [k in key_set for k in zip(*(df[c] for c in KEY))]
    return df[mask].copy()


# configurations admissible for a target security level: at least that secure
def at_least(df: pd.DataFrame, sec: int) -> pd.DataFrame:
    return df[df["security"] >= sec]


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[pareto_mask(df, ["runtime", "max_err"])].copy()


def _objectives(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["runtime"].to_numpy(dtype=float),
        np.log10(df["max_err"].to_numpy(dtype=float).clip(min=1e-300)),
    ])


# hypervolume of a set of 2D points (minimization) with respect to a reference point
def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))] # by x, then y

    hv = 0.0
    best_y = ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(hv)


# hypervolume ratio HV(tool) / HV(reference). Objectives are normalized by the ideal and nadir
# points of the REFERENCE front only, and the reference point is placed at 1 + margin in each
# normalized objective, slightly worse than the nadir (Ishibuchi et al. 2018). Normalizing by
# the reference front keeps the yardstick fixed: tool points worse than the reference point
# contribute nothing, and adding candidates to the tool can never lower its ratio.
#
# The runtime axis is normalized over at least runtime_tol of the fastest runtime: when a front
# lies within a few percent runtime (same N and chain length), its runtime differences are
# measurement noise and must not be stretched to the full axis.
def hypervolume_ratio(
    reference_front: pd.DataFrame,
    tool_front: pd.DataFrame,
    margin: float = 0.1,
    runtime_tol: float = 0.05,
) -> float:
    if reference_front.empty:
        return float("nan")
    if tool_front.empty:
        return 0.0

    ref_obj = _objectives(reference_front)
    tool_obj = _objectives(tool_front)

    ideal = ref_obj.min(axis=0)
    span = ref_obj.max(axis=0) - ideal
    span[0] = max(span[0], runtime_tol * ideal[0])
    span[span == 0] = 1.0

    ref_point = np.full(2, 1.0 + margin)
    hv_ref = hypervolume_2d((ref_obj - ideal) / span, ref_point)
    hv_tool = hypervolume_2d((tool_obj - ideal) / span, ref_point)
    return hv_tool / hv_ref if hv_ref > 0 else float("nan")


# fraction of reference-front configurations that the tool front contains exactly
def pareto_recall(reference_front: pd.DataFrame, tool_front: pd.DataFrame) -> float:
    if reference_front.empty:
        return float("nan")
    return len(keys(reference_front) & keys(tool_front)) / len(keys(reference_front))


# fraction of reference-front points that are covered by a tool-front point up to measurement
# noise: some tool point is at most (1 + runtime_tol) times slower and at most precision_tol_bits
# less precise. The tolerances should match the measured noise (see benchmark_suite/noise.py).
def epsilon_recall(
    reference_front: pd.DataFrame,
    tool_front: pd.DataFrame,
    runtime_tol: float = 0.05,
    precision_tol_bits: float = 1.5,
) -> float:
    if reference_front.empty:
        return float("nan")
    if tool_front.empty:
        return 0.0

    ref_obj = _objectives(reference_front)
    tool_obj = _objectives(tool_front)
    eps_log_err = precision_tol_bits * np.log10(2.0)

    covered = 0
    for rt, le in ref_obj:
        if np.any((tool_obj[:, 0] <= rt * (1 + runtime_tol)) & (tool_obj[:, 1] <= le + eps_log_err)):
            covered += 1
    return covered / len(ref_obj)


# multiplicative epsilon-indicator (Zitzler et al., IEEE TEC 2003): the smallest factor eps such
# that every reference-front point is weakly dominated by some tool-front point whose runtime and
# error are each multiplied by 1 / eps. eps = 1 means the tool front is as good as the reference
# front; eps = 1.05 means it is within 5% in both objectives (for error: within log2(1.05) bits).
def epsilon_indicator(reference_front: pd.DataFrame, tool_front: pd.DataFrame) -> float:
    if reference_front.empty:
        return float("nan")
    if tool_front.empty:
        return float("inf")

    ref = reference_front[["runtime", "max_err"]].to_numpy(dtype=float)
    tool = tool_front[["runtime", "max_err"]].to_numpy(dtype=float)
    # for each reference point b: min over tool points a of max_i a_i / b_i
    ratios = np.max(tool[None, :, :] / ref[:, None, :], axis=2)
    return float(np.max(np.min(ratios, axis=1)))


# agreement between the errors of the same configurations measured with two kinds of test
# vectors. Correlations are computed on log10(error).
def error_agreement(estimate: pd.Series, truth: pd.Series) -> dict:
    est = np.log10(np.asarray(estimate, dtype=float).clip(min=1e-300))
    tru = np.log10(np.asarray(truth, dtype=float).clip(min=1e-300))

    if len(est) < 3:
        nan = float("nan")
        return {"n": len(est), "spearman": nan, "pearson_log": nan, "r2_log": nan,
                "coverage": nan, "median_log_ratio": nan}

    ss_res = float(np.sum((tru - est) ** 2))
    ss_tot = float(np.sum((tru - tru.mean()) ** 2))

    return {
        "n": len(est),
        "spearman": float(spearmanr(est, tru).statistic),
        "pearson_log": float(pearsonr(est, tru).statistic),
        # R^2 of using the estimate directly as a prediction of the true log error
        "r2_log": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        # how often the estimate is at least the true error (a conservative estimate)
        "coverage": float(np.mean(est >= tru)),
        # median of log10(estimate / truth): > 0 overestimates, < 0 underestimates
        "median_log_ratio": float(np.median(est - tru)),
    }


def safe_ratio(a: float, b: float) -> float:
    return a / b if b and math.isfinite(b) and b != 0 else float("nan")


# compares a baseline configuration with every configuration the tool evaluated (all measured
# with the same test vectors). Returns the fastest configuration at least as precise as the
# baseline and the most precise configuration at least as fast, or None if there is none.
def dominating_configs(candidates: pd.DataFrame, baseline_err: float, baseline_runtime: float):
    as_precise = candidates[candidates["max_err"] <= baseline_err]
    as_fast = candidates[candidates["runtime"] <= baseline_runtime]
    fastest = as_precise.loc[as_precise["runtime"].idxmin()] if len(as_precise) else None
    most_precise = as_fast.loc[as_fast["max_err"].idxmin()] if len(as_fast) else None
    return fastest, most_precise
