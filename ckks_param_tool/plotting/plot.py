import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ckks_param_tool.config import PlottingConfig

'''
This file contains the plotting logic, one function ofr the plots for each security level 
and one for the combined plot.
'''

# the combined plot
def _requirement_line(ax, required_bits):
    if required_bits is not None:
        ax.axvline(2.0 ** -required_bits, color="gray", linestyle="--", linewidth=1, zorder=0)
        ax.annotate(f"needed: {required_bits} bits", (2.0 ** -required_bits, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=9, color="gray")


def plot_runtime_vs_precision_pareto(
    df: pd.DataFrame,
    config: PlottingConfig,
    required_bits: int | None = None,
):
    df_plot = df.copy()

    # if there is a user-defined error tolerance
    if config.error_tol is not None:
        df_plot = df_plot[df_plot["max_err"] < config.error_tol]

    # verifying that pareto logic has been applied
    if "is_pareto" not in df_plot or "is_recommended" not in df_plot:
        raise ValueError("DataFrame must contain 'is_pareto' and 'is_recommended' columns")

    df_plot = df_plot[df_plot["status"] == "success"]

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(9, 7))

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
            s=120,
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
            linewidths=2,
            s=150,
            zorder=3,
        )

        # plotting the recommended point
        recommended = level[level[f"recommended_{sec}"]]
        ax.scatter(
            recommended["max_err"],
            recommended["runtime"],
            marker="*",
            s=150,
            color=color,
            linewidths=2,
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
        marker="x",
        s=150,
        color="gray",
        linewidths=4,
        label="Recommended",
        zorder=5,
    )

    # proxy artist for Pareto-optimal point visualization
    pareto_proxy = ax.scatter(
        [], [],             
        facecolors="white",
        edgecolors="gray",
        s=150,
        linewidths=2,
        label="Optimal",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlabel("Maximum Absolute Error")
    ax.set_ylabel("Median Runtime (s)")
    ax.set_title("Runtime-Precision Tradeoff (All Security Levels)", pad=35)
    _requirement_line(ax, required_bits)

    sns.despine()

    # proxy artists for security level visualization
    security_proxies = [
        ax.scatter([], [], color=color, marker='s', s=120, alpha=0.35)
        for sec, color in palette.items()
    ]

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