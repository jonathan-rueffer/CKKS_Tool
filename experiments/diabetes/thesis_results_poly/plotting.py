# import pandas as pd

# df = pd.read_csv("experiments/diabetes/thesis_results_poly/Depth6/param_search.csv")

# avg_runtime = (
#     df.groupby("security")["runtime"]
#     .mean()
#     .reset_index()
# )

# print(avg_runtime)


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # --- Styling ---
# sns.set(style="ticks")
# plt.rcParams.update({
#     'figure.figsize': (9, 7),
#     'font.size': 14,
# })

# # --- Folder containing all depth CSVs ---
# base_dir = Path("experiments/diabetes/thesis_results_poly")
# depth_folders = sorted(base_dir.glob("Depth*"))

# all_data = []

# # --- Loop through depths ---
# for depth_folder in depth_folders:
#     depth = int(depth_folder.name.replace("Depth", ""))
#     csv_file = depth_folder / "param_search.csv"
    
#     df = pd.read_csv(csv_file)
#     df = df[df["runtime"] != float('inf')]  # remove inf runtimes
    
#     avg_runtime = df.groupby("security")["runtime"].mean().reset_index()
#     avg_runtime["depth"] = depth
#     all_data.append(avg_runtime)

# # --- Combine all depths ---
# plot_data = pd.concat(all_data, ignore_index=True)

# # --- Custom palette ---
# palette = {
#     128: "#F6A600",  # Orange
#     192: "#1B9E77",  # Green
#     256: "#2A6FDB",  # Blue
# }

# # --- Plot ---
# plt.figure(figsize=(10, 6))
# ax = sns.lineplot(
#     data=plot_data,
#     x="depth",
#     y="runtime",
#     hue="security",
#     marker="o",
#     palette=palette,
#     linewidth=2.5,
#     markersize=0
# )

# # --- Adjust y-axis to remove awkward gap ---
# plt.ylim(bottom=0)
# x_min = plot_data["depth"].min()
# x_max = plot_data["depth"].max()
# x_padding = 0.1  # or whatever fraction of the range you like
# plt.xlim(x_min - x_padding, x_max + x_padding)

# # --- Labels and title ---
# #plt.title("Average Runtime vs Depth (by Security Level)", fontsize=16, weight='bold')
# plt.xlabel("Depth")
# plt.ylabel("Average Runtime (s)")

# # --- Legend styling ---
# handles, labels = ax.get_legend_handles_labels()
# labels = [f"{str(int(float(lbl)))}-bit" for lbl in labels]
# ax.legend(handles, labels, fontsize=12, loc='upper left')
# ax.minorticks_off()

# sns.despine()

# # --- Tight layout ---
# plt.tight_layout()

# # --- Save and show ---
# plt.savefig("average_runtime_vs_depth.png", dpi=300)
# plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Styling ---
sns.set(style="ticks")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 14,
})

# --- Manual data: one runtime per depth ---
depths = [1, 2, 3, 4, 5, 6]
runtimes = [1.529, 2.401, 3.449, 4.762, 6.446, 11.579]  # replace with your actual numbers

df = pd.DataFrame({
    "Depth": depths,
    "Runtime (m)": runtimes
})

# --- Plot ---
ax = sns.lineplot(
    data=df,
    x="Depth",
    y="Runtime (m)",
    marker="s",
    color="#2A6FDB",  # Blue, thesis-quality
    linewidth=2.5,
    markersize=8
)

# --- Axes limits and padding ---
plt.ylim(bottom=0)
x_min = df["Depth"].min()
x_max = df["Depth"].max()
x_padding = 0.1
plt.xlim(x_min - x_padding, x_max + x_padding)

# --- Labels and title ---
#plt.title("Tool Runtime vs Depth", fontsize=16, weight='bold')
plt.xlabel("Depth")
plt.ylabel("Runtime (m)")
ax.minorticks_off()

sns.despine()

# --- Tight layout ---
plt.tight_layout()

# --- Save and show ---
plt.savefig("total_tool_runtime_vs_depth.png", dpi=300)
plt.show()





# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # --- Styling ---
# sns.set(style="ticks")
# plt.rcParams.update({
#     'figure.figsize': (9, 7),
#     'font.size': 14,
# })

# # --- Folder containing all depth CSVs ---
# base_dir = Path("experiments/diabetes/thesis_results_poly")
# depth_folders = sorted(base_dir.glob("Depth*"))

# all_data = []

# # --- Loop through depths ---
# for depth_folder in depth_folders:
#     depth = int(depth_folder.name.replace("Depth", ""))
#     csv_file = depth_folder / "param_search.csv"
    
#     df = pd.read_csv(csv_file)

#     # Remove invalid entries
#     df = df[(df["max_err"] != float('inf')) & (df["max_err"] > 0)]

#     avg_error = df.groupby("security")["max_err"].median().reset_index()
#     avg_error["depth"] = depth
#     all_data.append(avg_error)

# # --- Combine all depths ---
# plot_data = pd.concat(all_data, ignore_index=True)

# # --- Custom palette ---
# palette = {
#     128: "#F6A600",  # Orange
#     192: "#1B9E77",  # Green
#     256: "#2A6FDB",  # Blue
# }

# # --- Plot ---
# plt.figure(figsize=(10, 6))
# ax = sns.lineplot(
#     data=plot_data,
#     x="depth",
#     y="max_err",
#     hue="security",
#     marker="o",
#     palette=palette,
#     linewidth=2.5,
#     markersize=0
# )

# # --- Axis padding ---
# y_min = plot_data["max_err"].min()
# y_max = plot_data["max_err"].max()

# plt.ylim(y_min * 0.5, y_max * 2.3)

# x_min = plot_data["depth"].min()
# x_max = plot_data["depth"].max()
# x_padding = 0.1
# plt.xlim(x_min - x_padding, x_max + x_padding)

# plt.yscale("log")

# # --- Labels and title ---
# #plt.title("Precision vs Depth (by Security Level)", fontsize=16, weight='bold')
# plt.xlabel("Depth")
# plt.ylabel("Median Error")

# # --- Legend styling ---
# handles, labels = ax.get_legend_handles_labels()
# labels = [f"{str(int(float(lbl)))}-bit" for lbl in labels]
# #ax.legend(handles, labels, title="Security Level (bits)", title_fontsize=12, fontsize=12, loc='lower left')
# ax.legend(handles, labels, fontsize=12, loc='lower left')
# ax.minorticks_off()

# sns.despine()


# # --- Tight layout ---
# plt.tight_layout()

# # --- Save and show ---
# plt.savefig("median_error_vs_depth.png", dpi=300)
# plt.show()