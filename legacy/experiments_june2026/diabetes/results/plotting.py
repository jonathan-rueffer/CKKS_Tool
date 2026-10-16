import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Styling ---
sns.set(style="ticks")
plt.rcParams.update({
    "figure.figsize": (9, 7),
    "font.size": 13,
})

# --- Manually enter your results here ---
data = {
    "Truncated Normal": [0.258, 0.557, 0.203],   # example values
    "Gaussian": [0.117, 0.314, 0.349],
    "Uniform": [0.446, 0.590, 0.528],
    "Student-t": [0.539, 0.248, 0.259],
}

# --- Convert to long format ---
rows = []
for method, errors in data.items():
    for e in errors:
        rows.append({"method": method, "relative_error": e})

df = pd.DataFrame(rows)

# --- Plot ---
plt.figure()

sns.boxplot(
    data=df,
    x="method",
    y="relative_error",
    width=0.55,
    linewidth=1.5,
    color="#A6CEE3"
)

sns.despine()

plt.grid(False)
plt.tick_params(axis='x', length=0, pad=8)

plt.xlabel("Sampling Method")
plt.ylabel("Relative Error")
#plt.title("Accuracy of Representative Test Vector Sampling")

plt.tight_layout()
plt.savefig("test_vector_sampling_strat_vs_error.png", dpi=300)
plt.show()