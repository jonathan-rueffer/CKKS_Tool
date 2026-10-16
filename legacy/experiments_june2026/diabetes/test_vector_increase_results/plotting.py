import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Styling ---
sns.set_style("ticks")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 14,
})

# --- Folders for number of test vectors ---
folders = ["n5", "n10", "n15", "n20", "n25", "n30", "n35", "n40", "n45", "n50", "n55", "n60", "n65", "n70"]
base_dir = Path("experiments/diabetes/test_vector_increase_results")  # change this to your root folder

results = []

for folder_name in folders:
    folder_path = base_dir / folder_name
    n_vectors = int(folder_name[1:])  # extract number from 'n1', 'n5', etc.

    # --- Load recommended params ---
    rec_file = folder_path / "recommended_params.json"
    with open(rec_file, "r") as f:
        rec_params = json.load(f)

    # --- Load param_search.csv ---
    csv_file = folder_path / "param_search.csv"
    df = pd.read_csv(csv_file)
    
    # Remove failed experiments
    df = df[df["status"] == "success"]

    # --- Compute relative errors for all recommended params ---
    rel_errors = []
    for rec in rec_params:
        sec = rec["security"]
        poly = rec["poly_degree"]
        chain = rec["chain"]

        # Match rows in CSV
        df_match = df[
            (df["security"] == sec) &
            (df["poly_degree"] == poly) &
            (df["chain"].apply(lambda x: eval(x) == tuple(chain)))
        ]

        if df_match.empty:
            continue

        rec_err = rec["inference"]["error"]
        real_err = df_match["max_err"].median()  # or mean
        rel_error = abs(rec_err - real_err) / max(real_err, 1e-20)
        rel_errors.append(rel_error)

    if rel_errors:  # aggregate across security levels
        results.append({
            "n_test_vectors": n_vectors,
            "relative_error": np.median(rel_errors)  # or np.mean(rel_errors)
        })

# --- Convert to DataFrame ---
plot_df = pd.DataFrame(results).sort_values("n_test_vectors")

# --- Plot ---
plt.figure(figsize=(9,7))
sns.lineplot(
    data=plot_df,
    x="n_test_vectors",
    y="relative_error",
    marker="o",
    linewidth=2.5,
    markersize=0,
    color="#2A6FDB"
)
sns.despine()
#plt.yscale("log")  # relative errors are small
plt.xlabel("Number of Test Vectors")
plt.ylabel("Median Relative Error")
#plt.title("Relative Error vs Number of Test Vectors", weight="bold")

plt.xticks(plot_df["n_test_vectors"])
#plt.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("number_test_vectors_vs_error.png", dpi=300)
plt.show()