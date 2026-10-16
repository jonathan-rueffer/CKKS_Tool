import numpy as np
import pandas as pd

'''
This file contains the functions that define the Pareto frontier logic.
'''

# this function computes a dataframe mask that identifies the pareto optimal parameters
# (all metrics are minimized)
def pareto_mask(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.Series:

    values = df[metrics].to_numpy(dtype=float)
    n = values.shape[0]

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        # defines the dominate relation: no worse in every metric, strictly better in one
        dominates = (
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        dominates[i] = False

        if np.any(dominates):
            is_pareto[i] = False

    return pd.Series(is_pareto, index=df.index)

# this is a wrapper function that returns the pareto optimal rows of the results dataframe
def pareto_rows(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    mask = pareto_mask(df, metrics)
    return df[mask].copy()

def _bits(df: pd.DataFrame) -> pd.Series:
    return -np.log2(df["max_err"].astype(float).clip(lower=1e-300))


# the fastest front point; points at most runtime_tol slower count as equally fast, and the most
# precise of them is returned, so a noise-level speed difference never costs precision
def fastest_point(front: pd.DataFrame, runtime_tol: float = 0.05) -> pd.Series:
    fastest = float(front["runtime"].min())
    near = front[front["runtime"] <= fastest * (1 + runtime_tol)]
    return near.loc[near["max_err"].idxmin()]


# the most precise front point
def most_precise_point(front: pd.DataFrame) -> pd.Series:
    return front.loc[front["max_err"].idxmin()]


# the fastest front point whose precision (-log2 error) is at least required_bits (with the
# same runtime tolerance), or None if no point is precise enough
def fastest_meeting(front: pd.DataFrame, required_bits: float, runtime_tol: float = 0.05) -> pd.Series | None:
    meeting = front[_bits(front) >= required_bits]
    return fastest_point(meeting, runtime_tol) if len(meeting) else None


# this function selects the "knee" point of the frontier to recommend, but this can be changed if
# a given objective is deemed more important, such as error or runtime, through a 'policy' parameter.
# The knee is the point closest (Euclidean distance) to the ideal point after normalizing runtime
# and log10(error) to [0, 1]; error is compared on a log scale because it spans many orders
# of magnitude across scales.
#
# Before the knee is computed, runtimes within runtime_tol (relative) of each other are treated
# as equal: a front point is dropped if another front point is more accurate and at most
# runtime_tol slower. Otherwise a runtime difference at the level of measurement noise would be
# stretched to the full normalized range and could cost many bits of precision.
def select_representative_pareto(
    pareto_df: pd.DataFrame,
    policy: str = "knee",
    runtime_tol: float = 0.05,
) -> pd.Series:

    if len(pareto_df) == 1:
        return pareto_df.iloc[0]

    if policy == "fastest":
        return fastest_point(pareto_df, runtime_tol)

    if policy == "most_accurate":
        return most_precise_point(pareto_df)

    if policy != "knee":
        raise ValueError(f"Unknown recommendation policy '{policy}'")

    rt_all = pareto_df["runtime"].to_numpy(dtype=float)
    err_all = pareto_df["max_err"].to_numpy(dtype=float)
    keep = [
        not np.any((err_all < err_all[i]) & (rt_all <= rt_all[i] * (1 + runtime_tol)))
        for i in range(len(pareto_df))
    ]
    pareto_df = pareto_df[keep]

    if len(pareto_df) == 1:
        return pareto_df.iloc[0]

    rt = pareto_df["runtime"].astype(float)
    err = np.log10(pareto_df["max_err"].astype(float).clip(lower=1e-300))

    if rt.max() == rt.min() or err.max() == err.min():
        return pareto_df.loc[rt.idxmin()]

    rt_norm = (rt - rt.min()) / (rt.max() - rt.min())
    err_norm = (err - err.min()) / (err.max() - err.min())

    score = np.sqrt(rt_norm**2 + err_norm**2)

    # ties (always the case for a two-point front) go to the faster, then the more accurate
    # point, so the result does not depend on the row order
    order = pd.DataFrame({"score": score.round(12), "runtime": rt, "err": err})
    return pareto_df.loc[order.sort_values(["score", "runtime", "err"]).index[0]]
