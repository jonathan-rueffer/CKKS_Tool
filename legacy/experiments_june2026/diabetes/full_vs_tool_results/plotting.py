import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

base = Path("experiments/diabetes/full_vs_tool_results")  # change if needed

full_dir = base / "Fuller"
tool_dir = base / "Tool"

# --- Load full search ---
df_full = pd.read_csv(full_dir / "param_search.csv")
df_full = df_full[df_full["status"] == "success"]

# --- Load tool search ---
df_tool = pd.read_csv(tool_dir / "param_search.csv")
df_tool = df_tool[df_tool["status"] == "success"]

# --- Load tool recommended parameters ---
with open(tool_dir / "recommended_params.json") as f:
    rec_params = json.load(f)

for rec in rec_params:

    sec = rec["security"]
    poly = rec["poly_degree"]
    chain = rec["chain"]

    df_match = df_tool[
        (df_tool["security"] == sec) &
        (df_tool["poly_degree"] == poly) &
        (df_tool["chain"].apply(lambda x: eval(x) == tuple(chain)))
    ]

    if df_match.empty:
        continue

    rec["inference"]["error"] = df_match["max_err"].median()
    rec["inference"]["runtime"] = df_match["runtime"].iloc[0]


rec_runtime = [r["inference"]["runtime"] for r in rec_params]
rec_error = [r["inference"]["error"] for r in rec_params]

# --- Plot ---
plt.figure(figsize=(9,7))

# Full search space
plt.scatter(
    df_full["max_err"],
    df_full["runtime"],
    color="lightgray",
    alpha=0.4,
    s=40,
    label="Full Parameter Space"
)

# Tool explored configurations
plt.scatter(
    df_tool["max_err"],
    df_tool["runtime"],
    color="#1B9E77",
    alpha=0.7,
    s=50,
    label="Tool Evaluated Parameters"
)

# Tool final recommendations
plt.scatter(
    rec_error,
    rec_runtime,
    color="#D62728",
    marker="*",
    s=250,
    edgecolor="black",
    label="Tool Recommended Parameters"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Maximum Absolute Error")
plt.ylabel("Average Runtime (s)")
#plt.title("Full Parameter Space vs Tool Exploration", weight="bold")

sns.despine()
plt.minorticks_off()

plt.legend()

plt.tight_layout()
plt.savefig("full_space_vs_tool.png", dpi=300)
plt.show()