import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from ckks_param_tool.config.experiment import PlottingConfig
from ckks_param_tool.plotting import plot_runtime_vs_precision_pareto

from .analyze import latest_run
from .metrics import load_param_search, successful, keys, restrict_to, at_least, pareto_front

'''
Publication version of the tool-vs-grid front plot for the abstract (LNCS column width):

    python -m benchmark_suite.paper_figures fronts --results results/final_v3 --dataset diabetes
    python -m benchmark_suite.paper_figures tradeoff --results results/final_v3 --dataset heart_disease \
        --out paper/fheorg2027/figures/heart_tradeoff

x = median latency (ms, log scale), y = precision -log2(max abs error) on the synthetic test vectors.
All points are the reference grid's measurements (the tool's candidates paired with the grid), the
same data the front metrics use. The plot is cropped to the decision-relevant region; the number of
grid configurations outside it is printed for the caption.
'''

GRID_GRAY = "0.55"
TOOL_GRAY = "0.25"


def _staircase(front, x_max):
    # best precision attainable within a latency budget: a step function over the front
    f = front.sort_values("runtime")
    x = list(1000 * f["runtime"]) + [x_max]
    y = list(f["precision_bits"]) + [f["precision_bits"].iloc[-1]]
    return x, y


def _baseline_point(grid, base):
    # the grid's measurement of the baseline configuration if the grid contains it (paired with the
    # rest of the plot), otherwise the baseline's own test-vector measurement
    key = (base["poly_degree"], 2 ** base["log_scale"], tuple(base["chain"]))
    match = restrict_to(grid, {key})
    if not match.empty:
        return 1000 * match["runtime"].iloc[0], match["precision_bits"].iloc[0], True
    tv = base["test_vectors"]
    return 1000 * tv["runtime"], tv["precision_bits"], False


def plot_fronts_paper(name: str, run: Path, out: Path, sec: int = 128) -> dict:
    full = at_least(successful(load_param_search(run / "broad_grid" / "param_search.csv")), sec)
    tool = successful(load_param_search(run / "tool" / "param_search.csv"))
    paired = restrict_to(full, keys(tool))
    ff, tf = pareto_front(full), pareto_front(paired)
    with open(run / "real_rows.json") as f:
        knee = next(p for p in json.load(f) if p["point"] == "knee" and p["target_security"] == sec)
    knee = restrict_to(full, {(knee["poly_degree"], 2 ** knee["log_scale"], tuple(knee["chain"]))}).iloc[0]
    with open(run / "baseline.json") as f:
        baselines = {b["baseline"]: _baseline_point(full, b) for b in json.load(f)["baselines"]}

    x_max = 2 * 1000 * max(ff["runtime"].max(), tf["runtime"].max())
    x_min = 0.8 * 1000 * min(ff["runtime"].min(), tf["runtime"].min())
    y_max = max(ff["precision_bits"].max(), tf["precision_bits"].max()) + 2
    ms = 1000 * full["runtime"]
    inside = (ms <= x_max) & (ms >= x_min) & (full["precision_bits"] >= 0) & (full["precision_bits"] <= y_max)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax.scatter(ms, full["precision_bits"], s=3, color="lightgray", alpha=0.35, linewidths=0,
               label="grid configs", zorder=1)
    ax.scatter(1000 * paired["runtime"], paired["precision_bits"], s=6, color=TOOL_GRAY, linewidths=0,
               label="tool candidates", zorder=2)
    gx, gy = _staircase(ff, x_max)
    ax.step(gx, gy, where="post", color=GRID_GRAY, linewidth=1.5, label="grid front", zorder=3)
    ax.scatter(1000 * ff["runtime"], ff["precision_bits"], s=8, color=GRID_GRAY, linewidths=0, zorder=3)
    tx, ty = _staircase(tf, x_max)
    ax.step(tx, ty, where="post", color="black", linewidth=1.2, label="tool front", zorder=4)
    ax.scatter(1000 * tf["runtime"], tf["precision_bits"], s=6, color="black", linewidths=0, zorder=4)
    ax.scatter(1000 * knee["runtime"], knee["precision_bits"], s=40, facecolors="none", edgecolors="black",
               linewidths=1, label="knee", zorder=5)
    markers = {"tutorial": ("*", 50, "tutorial"), "expert": ("D", 18, "expert")}
    for base, (x, y, _) in baselines.items():
        marker, size, label = markers[base]
        ax.scatter(x, y, marker=marker, s=size, facecolors="white", edgecolors="black", linewidths=0.8,
                   label=label, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([t for t in (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80) if x_min <= t <= x_max])
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("Precision (bits)\n" r"$-\log_2$ max. abs. error")
    # legend above the axes, outside the data
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0, labelspacing=0.2, handlelength=1.2,
              fontsize=7)
    fig.tight_layout(pad=0.2)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{name}_fronts_{sec}_paper"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return {
        "png": str(stem.with_suffix(".png")),
        "grid_configurations": len(full),
        "outside_crop": int((~inside).sum()),
        "x_range_ms": (round(x_min, 2), round(x_max, 2)),
        "y_range_bits": (0, round(y_max, 1)),
        "baselines_from_grid": {b: v[2] for b, v in baselines.items()},
    }


# Fig. 2: the tool's own runtime-precision plot (all security levels) in paper style, with the
# reported configurations lettered as in the text: {(N, chain): letter or (letter, (dx, dy) points)}
TRADEOFF_LABELS = {
    "heart_disease": {
        (8192, (57, 50, 60)): "A",
        (8192, (20, 60, 60, 60)): "B",
        (8192, (52, 45, 52)): "C",
        (16384, (20, 60, 60, 60)): "D",
        (8192, (37, 30, 37)): "E",
        (16384, (57, 50, 60)): "F",
    },
}


def plot_tradeoff_paper(name: str, run: Path, out: Path) -> str:
    df = load_param_search(run / "tool" / "param_search.csv")
    with open(run / "tool" / "run_info.json") as f:
        required_bits = json.load(f).get("estimated_required_bits")
    fig = plot_runtime_vs_precision_pareto(df, PlottingConfig(), required_bits, style="paper",
                                           labels=TRADEOFF_LABELS.get(name))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return str(out.with_suffix(".png"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["fronts", "tradeoff"])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--dataset", default="diabetes")
    parser.add_argument("--security", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    run = latest_run(args.results / args.dataset)
    if args.figure == "fronts":
        print(plot_fronts_paper(args.dataset, run, args.out or Path("paper/fheorg2027/data/figures"), args.security))
    else:
        print(plot_tradeoff_paper(args.dataset, run, args.out or Path(f"paper/fheorg2027/figures/{args.dataset}_tradeoff")))


if __name__ == "__main__":
    main()
