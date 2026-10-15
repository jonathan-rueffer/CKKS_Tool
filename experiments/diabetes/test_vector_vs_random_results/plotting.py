import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("ticks")

plt.rcParams.update({
    "figure.figsize": (10,6),
    "font.size": 14,
})

base_dir = Path("experiments/diabetes/test_vector_vs_random_results")

results = []

for folder in base_dir.iterdir():

    name = folder.name

    if name.startswith("Random"):
        method = "Random"
        n_vectors = int(name.replace("Random",""))

    elif name.startswith("Representative"):
        method = "Representative"
        n_vectors = int(name.replace("Representative",""))

    else:
        continue

    rec_file = folder / "recommended_params.json"
    csv_file = folder / "param_search.csv"

    if not rec_file.exists():
        continue

    with open(rec_file) as f:
        rec_params = json.load(f)

    df = pd.read_csv(csv_file)
    df = df[df["status"] == "success"]
    

    for rec in rec_params:

        sec = rec["security"]
        poly = rec["poly_degree"]
        chain = rec["chain"]

        df_match = df[
            (df["security"] == sec) &
            (df["poly_degree"] == poly) &
            (df["chain"].apply(lambda x: eval(x) == tuple(chain)))
        ]

        if df_match.empty:
            continue

        rec_err = rec["inference"]["error"]
        real_err = df_match["max_err"].median()

        rel_error = abs(rec_err - real_err) / max(real_err, 1e-20)

        results.append({
            "Method": method,
            "n_vectors": n_vectors,
            "relative_error": rel_error
        })


plot_df = pd.DataFrame(results)

agg = plot_df.groupby(["Method","n_vectors"])["relative_error"].median().reset_index()

plt.figure()

sns.lineplot(
    data=agg,
    x="n_vectors",
    y="relative_error",
    hue="Method",
    marker="o",
    linewidth=2.5,
    palette={
        "Random":"#F6A600",
        "Representative":"#2A6FDB"
    },
    markersize=0
)

sns.despine()
plt.xlabel("Number of Test Vectors")
plt.ylabel("Median Relative Error")
#plt.title("Error Convergence of Test Vector Generation Strategies")
plt.legend(title=None)

plt.tight_layout()
plt.savefig("error_vs_test_vector_strat.png", dpi=300)
plt.show()