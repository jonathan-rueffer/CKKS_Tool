import ast
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

def plot_rep_vs_random_scatter(
    merged_df,
    dataset_name,
    save_path=None
):
    """
    Scatter plot comparing representative vs random errors
    for matched parameter configurations.
    """

    x = merged_df["max_err_rep"].to_numpy()
    y = merged_df["max_err_rand"].to_numpy()

    plt.figure(figsize=(6, 6))

    sns.scatterplot(
        x=x,
        y=y,
        alpha=0.7
    )

    # identity line (perfect agreement)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=1
    )

    plt.xlabel("Representative Test Vector Error")
    plt.ylabel("Random Test Vector Error")
    plt.title(f"{dataset_name}")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

# =========================
# Load / normalize
# =========================

def normalize(df):

    df = df.copy()

    df = df[df["status"] == "success"]

    df["max_err"] = pd.to_numeric(
        df["max_err"],
        errors="coerce"
    )

    df = df.dropna(subset=["max_err"])

    df["chain"] = df["chain"].apply(
        lambda x: tuple(ast.literal_eval(x))
    )

    return df


# =========================
# Parameter key
# =========================

def make_key(df):

    return list(
        zip(
            df["security"],
            df["poly_degree"],
            df["scale"],
            df["chain"]
        )
    )


# =========================
# Evaluate one dataset
# =========================

def evaluate(dataset_path):

    rep_runs = sorted(
        (dataset_path / "Representative").iterdir()
    )

    rand_runs = sorted(
        (dataset_path / "Random").iterdir()
    )

    rep_csv = rep_runs[-1] / "param_search.csv"
    rand_csv = rand_runs[-1] / "param_search.csv"

    rep_df = normalize(pd.read_csv(rep_csv))
    rand_df = normalize(pd.read_csv(rand_csv))


    rep_df["key"] = make_key(rep_df)
    rand_df["key"] = make_key(rand_df)

    merged = pd.merge(
        rep_df[["key", "max_err"]],
        rand_df[["key", "max_err"]],
        on="key",
        suffixes=("_rep", "_rand")
    )

    if len(merged) < 2:
        raise RuntimeError(
            f"Not enough matching configurations "
            f"for {dataset_path.name}"
        )

    x = merged["max_err_rep"].to_numpy()
    y = merged["max_err_rand"].to_numpy()

    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic

    r2 = r2_score(y, x)

    mae = mean_absolute_error(y, x)

    relative_mae = np.mean(
        np.abs(x - y) / np.maximum(np.abs(y), 1e-12)
    )

    base_save_dir = Path("benchmark_suite/test_vectors_vs_random_entries_experiments/scatter_plots")
    base_save_dir.mkdir(parents=True, exist_ok=True)

    save_path = base_save_dir / f"{dataset_path.name}_rep_vs_rand.png"

    plot_rep_vs_random_scatter(
        merged,
        dataset_path.name,
        save_path=str(save_path)
    )

    return pd.DataFrame([
        {
            "dataset": dataset_path.name,
            "n_configs": len(merged),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "r2": r2,
            "mae": mae,
            "relative_mae": relative_mae,
        }
    ])


# =========================
# Runner
# =========================

if __name__ == "__main__":

    base = Path(
        "benchmark_suite/"
        "test_vectors_vs_random_entries_experiments"
    )

    results = []

    for dataset_dir in sorted(base.iterdir()):

        if not dataset_dir.is_dir():
            continue

        if dataset_dir.name == "scatter_plots":
            continue 
    
        try:

            df = evaluate(dataset_dir)

            results.append(df)

            print(f"Done: {dataset_dir.name}")

        except Exception as e:

            print(
                f"Failed {dataset_dir.name}: {e}"
            )

    final_df = pd.concat(
        results,
        ignore_index=True
    )

    print("\n===== FINAL SUMMARY =====")
    print(final_df)

    # final_df.to_csv(
    #     "test_vector_analysis.csv",
    #     index=False
    # )

    print("\n===== LATEX =====")
    print(
        final_df.to_latex(
            index=False,
            float_format="%.4f"
        )
    )