import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ckks_param_tool.params import SECURITY_LEVELS

from .metrics import (
    KEY,
    load_param_search,
    successful,
    keys,
    restrict_to,
    at_least,
    pareto_front,
    pareto_recall,
    epsilon_recall,
    hypervolume_ratio,
    error_agreement,
    safe_ratio,
)

'''
Analyses benchmark runs produced by `python -m benchmark_suite.run` and writes the summary
tables (CSV and LaTeX) and figures to <results>/summary/.

Search efficiency (tool vs near-exhaustive "fuller" search)
    Every tool candidate is also a fuller candidate, so the tool's front is computed from the
    fuller run's measurements of the tool's candidates ("paired"). The comparison then reflects
    the search policy only, not run-to-run timing noise.

Test vectors
    The tool's candidates are evaluated with representative vectors (the tool), random dataset
    rows (the reference) and dataset-agnostic uniform vectors (a naive baseline).

Baseline
    Tutorial-default parameters versus the tool's recommendation, on the same real rows.
'''

logger = logging.getLogger("analyze")


def latest_run(dataset_dir: Path) -> Path | None:
    runs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and (p / "tool").exists())
    return runs[-1] if runs else None


def _run_info(path: Path) -> dict:
    with open(path / "run_info.json") as f:
        return json.load(f)


def search_efficiency(name: str, run: Path) -> list[dict]:
    if not (run / "fuller").exists():
        return []

    tool_all = load_param_search(run / "tool" / "param_search.csv")
    full_all = load_param_search(run / "fuller" / "param_search.csv")
    tool = successful(tool_all)
    full = successful(full_all)

    tool_keys = keys(tool)
    missing = len(tool_keys - keys(full_all))
    paired = restrict_to(full, tool_keys)

    tool_seconds = _run_info(run / "tool")["search_seconds"]
    full_seconds = _run_info(run / "fuller")["search_seconds"]

    rows = []
    for sec in SECURITY_LEVELS:
        full_s = at_least(full, sec)
        if full_s.empty:
            continue
        paired_s = at_least(paired, sec)
        tool_s = at_least(tool, sec)

        full_front = pareto_front(full_s)
        paired_front = pareto_front(paired_s)
        tool_front = pareto_front(tool_s)

        rows.append({
            "dataset": name,
            "security": sec,
            "tool_candidates": len(tool_s),
            "fuller_candidates": len(full_s),
            "candidate_reduction": 1 - safe_ratio(len(tool), len(full)),
            "search_time_reduction": 1 - safe_ratio(tool_seconds, full_seconds),
            "pareto_recall": pareto_recall(full_front, paired_front),
            "epsilon_recall": epsilon_recall(full_front, paired_front),
            "hypervolume_ratio": hypervolume_ratio(full_front, paired_front),
            # the same metrics with the tool's own (separately timed) measurements
            "hypervolume_ratio_unpaired": hypervolume_ratio(full_front, tool_front),
            "best_precision_bits_tool": float(paired_s["precision_bits"].max()) if len(paired_s) else np.nan,
            "best_precision_bits_fuller": float(full_s["precision_bits"].max()),
            "fastest_runtime_tool": float(paired_s["runtime"].min()) if len(paired_s) else np.nan,
            "fastest_runtime_fuller": float(full_s["runtime"].min()),
            "tool_keys_missing_from_fuller": missing,
        })
    return rows


def test_vector_agreement(name: str, run: Path) -> list[dict]:
    rep = successful(load_param_search(run / "tool" / "param_search.csv"))
    truth = successful(load_param_search(run / "test_vectors" / "random_rows.csv"))

    rows = []
    for label, path in (("representative", None), ("naive", run / "test_vectors" / "naive.csv")):
        est = rep if path is None else successful(load_param_search(path))
        merged = est[KEY + ["max_err"]].merge(truth[KEY + ["max_err"]], on=KEY, suffixes=("_est", "_true"))
        rows.append({
            "dataset": name,
            "test_vectors": label,
            **error_agreement(merged["max_err_est"], merged["max_err_true"]),
        })
    return rows


def baseline_comparison(name: str, run: Path) -> list[dict]:
    with open(run / "baseline.json") as f:
        data = json.load(f)

    base = data["baseline"]
    rows = []
    for rec in data["tool"]:
        rows.append({
            "dataset": name,
            "target_security": rec["target_security"],
            "tool_N": rec["poly_degree"],
            "tool_chain": tuple(rec["chain"]),
            "tool_security": rec["security"],
            "tool_precision_bits": rec["inference"]["precision_bits"],
            "tool_runtime": rec["inference"]["runtime"],
            "baseline_N": base["poly_degree"],
            "baseline_chain": tuple(base["chain"]),
            "baseline_security": base["security"],
            "baseline_precision_bits": base["inference"]["precision_bits"],
            "baseline_runtime": base["inference"]["runtime"],
            "speedup_vs_baseline": safe_ratio(base["inference"]["runtime"], rec["inference"]["runtime"]),
        })
    return rows


def plot_test_vectors(name: str, run: Path, out: Path):
    rep = successful(load_param_search(run / "tool" / "param_search.csv"))
    truth = successful(load_param_search(run / "test_vectors" / "random_rows.csv"))
    naive = successful(load_param_search(run / "test_vectors" / "naive.csv"))

    fig, ax = plt.subplots(figsize=(5, 5))
    for label, est, marker in (("representative", rep, "o"), ("naive", naive, "x")):
        m = est[KEY + ["max_err"]].merge(truth[KEY + ["max_err"]], on=KEY, suffixes=("_est", "_true"))
        ax.scatter(m["max_err_true"], m["max_err_est"], marker=marker, alpha=0.7, label=label)

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Max. error on random dataset rows")
    ax.set_ylabel("Max. error on test vectors")
    ax.set_title(name)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{name}_test_vectors.png", dpi=200)
    plt.close(fig)


def plot_fronts(name: str, run: Path, out: Path, sec: int = 128):
    if not (run / "fuller").exists():
        return
    full = at_least(successful(load_param_search(run / "fuller" / "param_search.csv")), sec)
    tool = successful(load_param_search(run / "tool" / "param_search.csv"))
    paired = restrict_to(full, keys(tool))
    if full.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(full["max_err"], full["runtime"], color="lightgray", s=15, label="fuller candidates")
    ax.scatter(paired["max_err"], paired["runtime"], color="tab:blue", s=15, alpha=0.5, label="tool candidates")
    ff, tf = pareto_front(full), pareto_front(paired)
    ax.scatter(ff["max_err"], ff["runtime"], facecolors="none", edgecolors="black", s=70, label="fuller front")
    ax.scatter(tf["max_err"], tf["runtime"], marker="x", color="tab:red", s=50, label="tool front")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum absolute error")
    ax.set_ylabel("Median runtime (s)")
    ax.set_title(f"{name} (at least {sec}-bit)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_fronts_{sec}.png", dpi=200)
    plt.close(fig)


def _write(df: pd.DataFrame, out: Path, stem: str):
    df.to_csv(out / f"{stem}.csv", index=False)
    with open(out / f"{stem}.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise benchmark runs.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    out = args.results / "summary"
    out.mkdir(parents=True, exist_ok=True)

    efficiency, vectors, baseline = [], [], []
    for dataset_dir in sorted(p for p in args.results.iterdir() if p.is_dir() and p.name != "summary"):
        run = latest_run(dataset_dir)
        if run is None:
            continue
        name = dataset_dir.name
        logger.info("analysing %s (%s)", name, run.name)

        efficiency += search_efficiency(name, run)
        vectors += test_vector_agreement(name, run)
        baseline += baseline_comparison(name, run)
        plot_test_vectors(name, run, out)
        plot_fronts(name, run, out)

    for rows, stem in ((efficiency, "search_efficiency"), (vectors, "test_vectors"), (baseline, "baseline")):
        if rows:
            df = pd.DataFrame(rows)
            _write(df, out, stem)
            print(f"\n===== {stem} =====")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
