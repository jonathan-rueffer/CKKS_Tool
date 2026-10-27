import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter, NullLocator, ScalarFormatter
import seaborn as sns
from ckks_param_tool.config import PlottingConfig

'''
This file contains the plotting logic, one function ofr the plots for each security level 
and one for the combined plot.
'''

# marker sizes and fonts: "default" for the tool's own output, "paper" for a single LNCS column
STYLES = {
    "default": dict(figsize=(9, 7), candidate=120, pareto=150, pareto_width=2, recommended=150, font=None,
                    legend_font=None),
    "paper": dict(figsize=(3.1, 2.5), candidate=14, pareto=24, pareto_width=1, recommended=48, font=8,
                  legend_font=7),
}


def _requirement_line(ax, required_bits, style="default"):
    if required_bits is not None:
        ax.axvline(2.0 ** -required_bits, color="gray", linestyle="--", linewidth=1 if style == "default" else 0.8,
                   zorder=0)
        label = f"needed: {required_bits} bits" if style == "default" else "needed (est.)"
        ax.annotate(label, (2.0 ** -required_bits, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -12 if style == "default" else -9), textcoords="offset points", ha="right",
                    fontsize=9 if style == "default" else 7, color="gray")


# paper style: latency in ms on a log axis, a top axis with precision in bits, bold letters
# next to the given configurations ({(poly_degree, chain): letter or (letter, (dx, dy) points)})
def _paper_axes(ax, df_plot, labels):
    ax.set_yscale("log")
    ax.set_yticks([3, 5, 10, 20, 30])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Latency (ms)", fontsize=8)
    ax.set_xlabel("Max. absolute error", fontsize=8)
    ax.tick_params(labelsize=8)

    top = ax.secondary_xaxis("top", functions=(lambda e: -np.log2(np.clip(e, 1e-300, None)), lambda b: 2.0 ** -b))
    # the secondary axis inherits the log scale; fix the bit ticks and their plain formatting
    top.xaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40]))
    top.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("Precision (bits)", fontsize=8)
    top.tick_params(labelsize=8)

    for (poly_degree, chain), label in (labels or {}).items():
        letter, offset = (label, (3, 3)) if isinstance(label, str) else label
        point = df_plot[(df_plot["poly_degree"] == poly_degree) & (df_plot["chain"].apply(tuple) == tuple(chain))]
        if point.empty:
            raise ValueError(f"no successful configuration N={poly_degree}, chain={chain} to label {letter}")
        ax.annotate(letter, (point["max_err"].iloc[0], point["runtime"].iloc[0]), xytext=offset,
                    textcoords="offset points", fontsize=7, fontweight="bold", zorder=6)


# the combined plot
def plot_runtime_vs_precision_pareto(
    df: pd.DataFrame,
    config: PlottingConfig,
    required_bits: int | None = None,
    style: str = "default",
    labels: dict | None = None,
):
    df_plot = df.copy()

    # if there is a user-defined error tolerance
    if config.error_tol is not None:
        df_plot = df_plot[df_plot["max_err"] < config.error_tol]

    # verifying that pareto logic has been applied
    if "is_pareto" not in df_plot or "is_recommended" not in df_plot:
        raise ValueError("DataFrame must contain 'is_pareto' and 'is_recommended' columns")

    df_plot = df_plot[df_plot["status"] == "success"]
    st = STYLES[style]
    if style == "paper":
        df_plot = df_plot.assign(runtime=1000 * df_plot["runtime"])

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=st["figsize"])

    palette = {
        128: "#F6A600",  
        192: "#1B9E77",  
        256: "#2A6FDB",  
    }

    for sec, color in palette.items():

        # candidates are coloured by the highest security level they reach
        subset = df_plot[df_plot["security"] == sec]

        # plotting the candidate points
        candidates = subset[~subset["is_recommended"]]
        ax.scatter(
            candidates["max_err"],
            candidates["runtime"],
            color=color,
            alpha=0.35,
            s=st["candidate"],
            edgecolors="none",
            zorder=1,
        )

        # the front of a level contains every configuration at least that secure
        level = df_plot[df_plot["security"] >= sec]

        # plotting the pareto points (excluding the recommended one)
        pareto = level[(level[f"pareto_{sec}"]) & (~level[f"recommended_{sec}"])]
        ax.scatter(
            pareto["max_err"],
            pareto["runtime"],
            facecolors="white",
            edgecolors=color,
            linewidths=st["pareto_width"],
            s=st["pareto"],
            zorder=3,
        )

        # plotting the recommended point
        recommended = level[level[f"recommended_{sec}"]]
        ax.scatter(
            recommended["max_err"],
            recommended["runtime"],
            marker="*",
            s=st["recommended"],
            color=color,
            linewidths=st["pareto_width"],
            label=f"{sec}-bit Recommended",
            zorder=5,
        )


    # proxy artist for candidate point visualization
    candidate_proxy = ax.scatter(
        [], [],            
        color="gray",
        alpha=0.35,
        s=120,
        edgecolors="none",
        label="Candidate",
        zorder=1,
    )

    # proxy artist for recommended point visualization
    recommended_proxy = ax.scatter(
        [], [],              
        marker="*",
        s=st["recommended"],
        color="gray",
        label="Recommended",
        zorder=5,
    )

    # proxy artist for Pareto-optimal point visualization
    pareto_proxy = ax.scatter(
        [], [],             
        facecolors="white",
        edgecolors="gray",
        s=st["pareto"],
        linewidths=st["pareto_width"],
        label="Optimal",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.minorticks_off()
    _requirement_line(ax, required_bits, style)
    sns.despine()

    # proxy artists for security level visualization
    security_proxies = [
        ax.scatter([], [], color=color, marker='s', s=st["candidate"] if style == "paper" else 120, alpha=0.35)
        for sec, color in palette.items()
    ]

    if style == "paper":
        _paper_axes(ax, df_plot, labels)
        # one row above the top axis, in figure coordinates so tight_layout leaves room for it
        fig.legend(
            handles=security_proxies + [pareto_proxy, recommended_proxy],
            labels=[f"{sec}-bit" for sec in palette.keys()] + ["Pareto-optimal", "Recommended"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(palette) + 2,
            frameon=False,
            fontsize=st["legend_font"],
            handlelength=0.8,
            handletextpad=0.25,
            columnspacing=0.7,
            borderaxespad=0.1,
        )
        fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.92))
        return fig

    ax.set_xlabel("Maximum Absolute Error")
    ax.set_ylabel("Median Runtime (s)")
    ax.set_title("Runtime-Precision Tradeoff (All Security Levels)", pad=35)

    # legend formatting
    ax.legend(
        handles=security_proxies + [candidate_proxy, pareto_proxy, recommended_proxy],
        labels=[f"{sec}-bit" for sec in palette.keys()] +  ["Candidate", "Optimal", "Recommended"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(palette)+3,
        frameon=False,
        handletextpad=0.1,
        columnspacing=1.1,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

# the individual security level plot
# shows every configuration that is at least `sec`-bit secure, with the front for that target
def plot_single_security_level(df: pd.DataFrame, sec: int, config: PlottingConfig,
                               required_bits: int | None = None):

    subset = df[(df["status"] == "success") & (df["security"] >= sec)].copy()

    if subset.empty:
        return None

    if config.error_tol is not None:
        subset = subset[subset["max_err"] < config.error_tol]

    sns.set_theme(style="ticks")

    fig, ax = plt.subplots(figsize=(9, 7))

    palette = {
        128: "#F6A600", 
        192: "#1B9E77",
        256: "#2A6FDB", 
    }
    main_color = palette[sec]  # get the right color for the security level

    # plotting all points (except for recommended for no visual overlap issues)
    all_points = subset[~subset[f"recommended_{sec}"]]
    ax.scatter(
        all_points["max_err"],
        all_points["runtime"],
        color=main_color,
        alpha=0.35,
        s=120,
        edgecolors="none",
        label="Candidate Parameters",
        zorder=1,
    )

    # plotting all Pareto-optimal candidate parameter sets
    pareto = subset[(subset[f"pareto_{sec}"]) & (~subset[f"recommended_{sec}"])]
    ax.scatter(
        pareto["max_err"],
        pareto["runtime"],
        facecolors="white",
        edgecolors=main_color,
        linewidths=2,
        s=150,
        label="Pareto-Optimal",
        zorder=3,
    )

    # plotting the recommended point
    recommended = subset[subset[f"recommended_{sec}"]]
    ax.scatter(
        recommended["max_err"],
        recommended["runtime"],
        marker="*",
        s=150,
        color=main_color, 
        linewidths=2,
        label="Recommended",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlabel("Maximum Absolute Error")
    ax.set_ylabel("Median Runtime (s)")
    ax.set_title(f"Runtime-Precision Tradeoff (at least {int(sec)}-bit Security)", pad=35)
    _requirement_line(ax, required_bits)

    sns.despine()

    # legend formatting
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig