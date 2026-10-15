import pandas as pd
import numpy as np
import json
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pathlib import Path

VALID_SECURITY_LEVELS = {128, 192, 256}


# =========================
# Parse chain safely
# =========================
def normalize(df):
    df = df.copy()

    df = df[df["status"] == "success"]

    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["max_err"] = pd.to_numeric(df["max_err"], errors="coerce")

    df = df.dropna(subset=["runtime", "max_err", "security"])

    # IMPORTANT: convert string "(50, 40, 60)" → tuple
    def parse_chain(x):
        if isinstance(x, tuple):
            return x
        return tuple(ast.literal_eval(x))

    df["chain"] = df["chain"].apply(parse_chain)

    return df

def pareto_mask(df: pd.DataFrame, metrics: list[str]) -> pd.Series:

    values = df[metrics].to_numpy(dtype=float)
    n = values.shape[0]

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        dominates = (
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        dominates[i] = False

        if np.any(dominates):
            is_pareto[i] = False

    return pd.Series(is_pareto, index=df.index)


# =========================
# Pareto frontier (WRAPPER)
# =========================
def pareto_frontier(df, metrics=("max_err", "runtime")):

    df = df.copy()

    # clean input (critical for stability)
    df = df.dropna(subset=list(metrics))

    if len(df) == 0:
        return df.iloc[0:0]

    mask = pareto_mask(df, list(metrics))

    pareto_df = df[mask].copy()

    return pareto_df

# =========================
# Split by security
# =========================
def split_by_security(df):
    df = df.copy()
    df = df[df["security"].isin(VALID_SECURITY_LEVELS)]

    return {
        int(s): df[df["security"] == s].copy()
        for s in sorted(df["security"].unique())
    }


# =========================
# KEY-BASED PARETO RECALL (CORRECT)
# =========================
def pareto_recall(full_pf, tool_pf):

    if len(full_pf) == 0 or len(tool_pf) == 0:
        return 0.0

    def make_key(df):
        return set(
            zip(
                df["poly_degree"],
                df["scale"],
                df["chain"]
            )
        )

    full_set = make_key(full_pf)
    tool_set = make_key(tool_pf)

    return len(full_set & tool_set) / len(full_set)

# =========================
# Hypervolume (2D minimization)
# =========================

def hypervolume_2d(points, ref, debug=False):
    """
    Computes 2D hypervolume for minimization objectives.

    Parameters
    ----------
    points : ndarray (n,2)
        Columns:
            x = max_err
            y = runtime

    ref : ndarray (2,)
        Reference point (must be worse than all points)

    Returns
    -------
    float
    """

    pts = np.asarray(points, dtype=float)

    if len(pts) == 0:
        return 0.0

    # remove NaNs
    pts = pts[~np.isnan(pts).any(axis=1)]

    if len(pts) == 0:
        return 0.0

    # remove duplicates
    pts = np.unique(pts, axis=0)

    # sort by error (ascending)
    pts = pts[np.argsort(pts[:, 0])]

    hv = 0.0

    current_y = ref[1]

    if debug:
        print("\n--- Hypervolume Calculation ---")
        print("Reference:", ref)
        print("Points:")
        print(pts)

    for x, y in pts:

        width = ref[0] - x
        height = current_y - y

        contrib = 0.0

        if width > 0 and height > 0:
            contrib = width * height
            hv += contrib

        if debug:
            print(
                f"x={x:.6e}, "
                f"y={y:.6e}, "
                f"width={width:.6e}, "
                f"height={height:.6e}, "
                f"contrib={contrib:.6e}"
            )

        current_y = min(current_y, y)

    if debug:
        print("HV =", hv)
        print("-------------------------------\n")

    return float(hv)


# =========================
# Debug helper
# =========================

def debug_frontiers(full_pf, tool_pf, ref):

    print("\n==================================================")
    print("FULL PARETO FRONTIER")
    print("==================================================")

    print(
        full_pf[
            [
                "poly_degree",
                "scale",
                "chain",
                "max_err",
                "runtime"
            ]
        ]
    )

    print("\n==================================================")
    print("TOOL PARETO FRONTIER")
    print("==================================================")

    print(
        tool_pf[
            [
                "poly_degree",
                "scale",
                "chain",
                "max_err",
                "runtime"
            ]
        ]
    )

    hv_full = hypervolume_2d(
        full_pf[["max_err", "runtime"]].to_numpy(),
        ref,
        debug=True
    )

    hv_tool = hypervolume_2d(
        tool_pf[["max_err", "runtime"]].to_numpy(),
        ref,
        debug=True
    )

    print("\n==================================================")
    print("HYPERVOLUME COMPARISON")
    print("==================================================")
    print("HV_FULL =", hv_full)
    print("HV_TOOL =", hv_tool)

    if hv_full > 0:
        print("RATIO =", hv_tool / hv_full)

    print("==================================================\n")



# =========================
# Core evaluation
# =========================
def evaluate(dataset_path):

    full_runs = sorted((dataset_path / "Fuller").iterdir())
    tool_runs = sorted((dataset_path / "Tool").iterdir())

    full_csv = full_runs[-1] / "param_search.csv"
    tool_csv = tool_runs[-1] / "param_search.csv"

    full_df = normalize(pd.read_csv(full_csv))
    tool_df = normalize(pd.read_csv(tool_csv))

    results = []

    full_by_sec = split_by_security(full_df)
    tool_by_sec = split_by_security(tool_df)

    for sec in sorted(full_by_sec.keys()):

        full_s = full_by_sec[sec]
        tool_s = tool_by_sec.get(sec, pd.DataFrame())

        if len(full_s) == 0:
            continue

        full_pf = pareto_frontier(full_s)
        tool_pf = pareto_frontier(tool_s)

        # -------------------------
        # candidate reduction
        # -------------------------
        cand_red = 1 - (len(tool_s) / len(full_s)) if len(full_s) > 0 else 0

        # -------------------------
        # runtime reduction
        # -------------------------
        full_rt = full_s["runtime"].mean()
        tool_rt = tool_s["runtime"].mean() if len(tool_s) > 0 else np.nan

        run_red = (
            1 - (tool_rt / full_rt)
            if full_rt > 0 and not np.isnan(tool_rt)
            else np.nan
        )

        # -------------------------
        # reference point
        # -------------------------
        ref = np.array([
            full_s["max_err"].max() * 1.1,
            full_s["runtime"].max() * 1.1
        ], dtype=float)

        hv_full = hypervolume_2d(
            full_pf[["max_err", "runtime"]].to_numpy(),
            ref
        )

        hv_tool = hypervolume_2d(
            tool_pf[["max_err", "runtime"]].to_numpy(),
            ref
        )

        results.append({
            "dataset": dataset_path.name,
            "security": sec,
            "candidate_reduction": cand_red,
            "runtime_reduction": run_red,
            "pareto_recall": pareto_recall(full_pf, tool_pf),
            "hypervolume_ratio": (hv_tool / hv_full) if hv_full > 0 else 0.0
        })

    return pd.DataFrame(results)


# =========================
# Runner
# =========================
if __name__ == "__main__":

    base = Path("benchmark_suite/full_vs_tool_experiments")

    all_results = []

    for dataset_dir in sorted(base.iterdir()):

        if not dataset_dir.is_dir():
            continue

        if not (dataset_dir / "Fuller").exists():
            continue

        try:
            df = evaluate(dataset_dir)
            all_results.append(df)
            print(f"Done: {dataset_dir.name}")

        except Exception as e:
            print(f"Failed {dataset_dir.name}: {e}")

    final_df = pd.concat(all_results, ignore_index=True)

    print("\n===== FINAL SUMMARY =====")
    print(final_df)

    final_df.to_csv("benchmark_summary.csv", index=False)

    print("\n===== LATEX =====")
    print(final_df.to_latex(index=False, float_format="%.4f"))



