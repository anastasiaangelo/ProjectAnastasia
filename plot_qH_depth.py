# %%  import
import pandas as pd
import matplotlib.pyplot as plt

# %% ################################# One curve per num rots per res ##########################################
# Load datasets
depth2_df = pd.read_csv("RESULTS/qH_depths/qH_depth_2rots_nopenalty.csv").rename(columns={"Depth": "Depth_2rots"})
depth3_df = pd.read_csv("RESULTS/qH_depths/qH_depth_3rots_nopenalty.csv").rename(columns={"Depth": "Depth_3rots"})
depth4_df = pd.read_csv("RESULTS/qH_depths/qH_depth_4rots_nopenalty.csv").rename(columns={"Depth": "Depth_4rots"})
depth5_df = pd.read_csv("RESULTS/qH_depths/qH_depth_5rots_nopenalty.csv").rename(columns={"Depth": "Depth_5rots"})
depth6_df = pd.read_csv("RESULTS/qH_depths/qH_depth_6rots_nopenalty.csv").rename(columns={"Depth": "Depth_6rots"})
depth7_df = pd.read_csv("RESULTS/qH_depths/qH_depth_7rots_nopenalty.csv").rename(columns={"Depth": "Depth_7rots"})

# Merge datasets on 'Size'
df = depth2_df
for depth_df in [depth3_df, depth4_df, depth5_df, depth6_df, depth7_df]:
    df = pd.merge(df, depth_df, on="Size", how="outer")

# Sort by Circuit Size
df = df.sort_values(by="Size")
df.interpolate(inplace=True)  # Fill missing values smoothly

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Plot depths for different rotamer residues
ax.plot(df["Size"], df["Depth_2rots"], marker='o', linestyle='-', label="Depth 2 rots", color='indianred')
ax.plot(df["Size"], df["Depth_3rots"], marker='s', linestyle='--', label="Depth 3 rots", color='royalblue')
ax.plot(df["Size"], df["Depth_4rots"], marker='x', linestyle='-.', label="Depth 4 rots", color='darkorange')
ax.plot(df["Size"], df["Depth_5rots"], marker='^', linestyle='--', label="Depth 5 rots", color='darkviolet')
ax.plot(df["Size"], df["Depth_6rots"], marker='d', linestyle='-', label="Depth 6 rots", color='seagreen')
ax.plot(df["Size"], df["Depth_7rots"], marker='P', linestyle='--', label="Depth 7 rots", color='palevioletred')

# Labels and title
ax.set_xlabel("Number of Qubits", fontsize=24)
ax.set_ylabel("Circuit Depth", fontsize=24)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.legend(fontsize=22, loc='lower right')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# Save the plot
pdf_filename = "Paper Plots/qH_circuit_depth_nrots.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")  # Ensures no clipping

plt.show()


# %% ################################# One curve for all values penalties ##########################################
# Load dataset
from scipy.optimize import curve_fit
import numpy as np
df = pd.read_csv("RESULTS/qH_depths/qH_depths_penalties.csv")
df2 = pd.read_csv("RESULTS/qH_depths/total_depths_penalties.csv")

# Sort by Circuit Size
df = df.sort_values(by="Size")
df.interpolate(inplace=True)  # Smooth missing values if necessary

df2 = df2.sort_values(by="Size")
df2.interpolate(inplace=True)  # Smooth missing values if necessary

# Logarithmic model
def log_fit(x, a, b):
    return a * np.log(x) + b

# Fit the power law to data
x_data = df["Size"].values
y_data = df["Depth"].values

x2_data = df2["Size"].values
y2_data = df2["Depth"].values

# Avoid fitting on zero/negative values
mask = (x_data > 0) & (y_data > 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
x_fit = x_data[mask]
y_fit = y_data[mask]

mask2 = (x2_data > 0) & (y2_data > 0) & (~np.isnan(x2_data)) & (~np.isnan(y2_data))
x2_fit = x2_data[mask2]
y2_fit = y2_data[mask2]

# Curve fitting
x_smooth = np.linspace(x_fit.min(), x_fit.max(), 200)
from sklearn.metrics import r2_score

# --- Fit logarithmic ---
popt_log, _ = curve_fit(log_fit, x_fit, y_fit, p0=(1, 1))
y_log_fit = log_fit(x_fit, *popt_log)
r2_log = r2_score(y_fit, y_log_fit)

popt2_log, _ = curve_fit(log_fit, x2_fit, y2_fit, p0=(1, 1))
y2_log_fit = log_fit(x2_fit, *popt2_log)
r2_log2 = r2_score(y2_fit, y2_log_fit)

# Create figure
fig, ax = plt.subplots(figsize=(11, 10))

# Plot raw lines
ax.plot(df["Size"], df["Depth"], marker='o', linestyle='-', color='lightgreen', label="Cost Hamiltonian")
ax.plot(df2["Size"], df2["Depth"], marker='s', linestyle='-', color='mediumseagreen', label="Total QAOA Layer")

# Scatter overlay
ax.scatter(df["Size"], df["Depth"], color='lightgreen', edgecolors='darkseagreen', zorder=3)
ax.scatter(df2["Size"], df2["Depth"], color='mediumseagreen', edgecolors='seagreen', zorder=3)

# Fitted curves
ax.plot(x_smooth, log_fit(x_smooth, *popt_log), '--', color='lightgreen', linewidth=1.5,
        label=f"Log Fit Cost")

ax.plot(x_smooth, log_fit(x_smooth, *popt2_log), '--', color='mediumseagreen', linewidth=1.5,
        label=f"Log Fit Total")


# Labels and title
ax.set_xlabel("Number of Qubits", fontsize=24)
ax.set_ylabel("Circuit Depth", fontsize=24)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.legend(fontsize=22, loc='lower right')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# Save the improved plot
pdf_filename = "Paper Plots/qH_circuit_depth_penalties.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")  # Ensures no clipping
plt.show()


# %% ################################# One curve for all values no penalties ##########################################
# Load dataset
from scipy.optimize import curve_fit
import numpy as np
df = pd.read_csv("RESULTS/qH_depths/qH_depths.csv")
df2 = pd.read_csv("RESULTS/qH_depths/total_depths.csv")

# Sort by Circuit Size
df = df.sort_values(by="Size")
df.interpolate(inplace=True)  # Smooth missing values if necessary

df2 = df2.sort_values(by="Size")
df2.interpolate(inplace=True) 

# Logarithmic model
def log_fit(x, a, b):
    return a * np.log(x) + b

# Fit the power law to data
x_data = df["Size"].values
y_data = df["Depth"].values

x2_data = df2["Size"].values
y2_data = df2["Depth"].values

# Avoid fitting on zero/negative values
mask = (x_data > 0) & (y_data > 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
x_fit = x_data[mask]
y_fit = y_data[mask]

mask2 = (x2_data > 0) & (y2_data > 0) & (~np.isnan(x2_data)) & (~np.isnan(y2_data))
x2_fit = x2_data[mask2]
y2_fit = y2_data[mask2]

# Curve fitting
x_smooth = np.linspace(x_fit.min(), x_fit.max(), 200)
from sklearn.metrics import r2_score

# --- Fit logarithmic ---
popt_log, _ = curve_fit(log_fit, x_fit, y_fit, p0=(1, 1))
y_log_fit = log_fit(x_fit, *popt_log)
r2_log = r2_score(y_fit, y_log_fit)

popt2_log, _ = curve_fit(log_fit, x2_fit, y2_fit, p0=(1, 1))
y2_log_fit = log_fit(x2_fit, *popt2_log)
r2_log2 = r2_score(y2_fit, y2_log_fit)

# Create figure
fig, ax = plt.subplots(figsize=(11, 10))

# Plot Depth as a line with markers for better clarity
ax.plot(df["Size"], df["Depth"], marker='o', linestyle='-', color='lightsteelblue', label="Cost Hamiltonian", alpha=0.8)
ax.plot(df2["Size"], df2["Depth"], marker='s', linestyle='-', color='royalblue', label="Total QAOA Layer", alpha=0.8)

# Add a scatter plot on top for better visualization
ax.scatter(df["Size"], df["Depth"], color='lightsteelblue', edgecolors='steelblue', zorder=3)
ax.scatter(df2["Size"], df2["Depth"], color='royalblue', edgecolors='blue', zorder=3)

# Plot fitted curve
ax.plot(x_smooth, log_fit(x_smooth, *popt_log), '--', color='steelblue', linewidth=1.5,
        label=f"Log Fit Cost")
ax.plot(x_smooth, log_fit(x_smooth, *popt2_log), '--', color='blue', linewidth=1.5,
        label=f"Log Fit Total")

handles, labels = ax.get_legend_handles_labels()
order = [0, 2, 1, 3]  
# Labels and title
ax.set_xlabel("Number of Qubits", fontsize=24)
ax.set_ylabel("Circuit Depth", fontsize=24)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    fontsize=22,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),  # adjust vertical position
    ncol=2,                        # <-- two columns
    columnspacing=1.5,
    handletextpad=0.5
)
# ax.legend(fontsize=22, loc='lower right')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# Save the improved plot
pdf_filename = "Paper Plots/qH_circuit_depth.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")  # Ensures no clipping
plt.show()

# %% ################################# Penalties vs. no penalties ##########################################
df1 = pd.read_csv("RESULTS/qH_depths/qH_depths_penalties.csv").rename(columns={"Depth": "Depth_with_penalty"})
df2 = pd.read_csv("RESULTS/qH_depths/qH_depths.csv").rename(columns={"Depth": "Depth_no_penalty"})
df3 = pd.read_csv("RESULTS/qH_depths/total_depths_penalties.csv").rename(columns={"Depth": "Total_depth_with_penalty"})
df4 = pd.read_csv("RESULTS/qH_depths/total_depths.csv").rename(columns={"Depth": "Total_depth_no_penalty"})

df = pd.merge(df1, df2, on="Size", how="outer")
df = pd.merge(df, df3, on="Size", how="outer")
df = pd.merge(df, df4, on="Size", how="outer")
df = df.sort_values(by="Size")

df.interpolate(inplace=True)  # Fill missing values smoothly

# Mask for valid data
mask_penalty = df["Size"].notna() & df["Depth_with_penalty"].notna() & (df["Size"] > 0)
mask_nopenalty = df["Size"].notna() & df["Depth_no_penalty"].notna() & (df["Size"] > 0)

mask_totalpenalty = df["Size"].notna() & df["Total_depth_with_penalty"].notna() & (df["Size"] > 0)
mask_totalnopenalty = df["Size"].notna() & df["Total_depth_no_penalty"].notna() & (df["Size"] > 0)


x_penalty = df["Size"][mask_penalty].values
y_penalty = df["Depth_with_penalty"][mask_penalty].values

x_nopenalty = df["Size"][mask_nopenalty].values
y_nopenalty = df["Depth_no_penalty"][mask_nopenalty].values

x_totpenalty = df["Size"][mask_totalpenalty].values
y_totpenalty = df["Total_depth_with_penalty"][mask_totalpenalty].values

x_totnopenalty = df["Size"][mask_totalnopenalty].values
y_totnopenalty = df["Total_depth_no_penalty"][mask_totalnopenalty].values

# Fit both
popt_penalty, _ = curve_fit(log_fit, x_penalty, y_penalty, p0=(1, 1))
popt_nopenalty, _ = curve_fit(log_fit, x_nopenalty, y_nopenalty, p0=(1, 1))
popt_totpenalty, _ = curve_fit(log_fit, x_totpenalty, y_totpenalty, p0=(1, 1))
popt_totnopenalty, _ = curve_fit(log_fit, x_totnopenalty, y_totnopenalty, p0=(1, 1))

# Smooth X for plotting
x_smooth = np.linspace(df["Size"].min(), df["Size"].max(), 300)

y_fit_penalty = log_fit(x_smooth, *popt_penalty)
y_fit_nopenalty = log_fit(x_smooth, *popt_nopenalty)
y_fit_totpenalty = log_fit(x_smooth, *popt_totpenalty)
y_fit_totnopenalty = log_fit(x_smooth, *popt_totnopenalty)

# Optionally compute R² values
r2_penalty = r2_score(y_penalty, log_fit(x_penalty, *popt_penalty))
r2_nopenalty = r2_score(y_nopenalty, log_fit(x_nopenalty, *popt_nopenalty))

fig, ax = plt.subplots(figsize=(11, 10))
plt.style.use('default') 
# === Cost Depth (With Penalty) ===
ax.plot(df["Size"], df["Depth_with_penalty"], linestyle='-', color='lightgreen', alpha=0.5, label="Depth Cost with Penalties")
ax.scatter(df["Size"], df["Depth_with_penalty"], marker='o', color='white', edgecolors='lightgreen', zorder=2)

# === Cost Depth (No Penalty) ===
ax.plot(df["Size"], df["Depth_no_penalty"], linestyle='--', color='lightsteelblue', alpha=0.5, label="Depth Cost No Penalties")
ax.scatter(df["Size"], df["Depth_no_penalty"], marker='o', facecolor='white', edgecolor='lightsteelblue', zorder=2)

# === Total Depth (With Penalty) ===
ax.plot(df["Size"], df["Total_depth_with_penalty"], linestyle='-', color='mediumseagreen', alpha=0.5, label="Total Depth with Penalties")
ax.scatter(df["Size"], df["Total_depth_with_penalty"], marker='s', color='white', edgecolors='mediumseagreen', zorder=2)

# === Total Depth (No Penalty) ===
ax.plot(df["Size"], df["Total_depth_no_penalty"], linestyle='--', color='royalblue', alpha=0.5, label="Total Depth No Penalties")
ax.scatter(df["Size"], df["Total_depth_no_penalty"], marker='s', facecolor='white', edgecolor='royalblue', zorder=2)

# === Log Fit Curves ===
ax.plot(x_smooth, y_fit_penalty, color='limegreen', linestyle='-', linewidth=1.8, label="Fit")
ax.plot(x_smooth, y_fit_nopenalty, color='#6699FF', linestyle='-', linewidth=1.8, label="Fit")
ax.plot(x_smooth, y_fit_totpenalty, color='forestgreen', linestyle='-', linewidth=1.8, label="Fit")
ax.plot(x_smooth, y_fit_totnopenalty, color='#3399FF', linestyle='-', linewidth=1.8, label="Fit")

# === Axes & Grid ===
ax.set_xlabel("Number of Qubits", fontsize=24)
ax.set_ylabel("Circuit Depth", fontsize=24)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.legend(fontsize=20, loc='lower right', ncol=2)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)


pdf_filename = "Paper Plots/qH_circuit_depth_penalties_vs_nopenalties_fit.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight") 

plt.show()


# %% #################################  num rots per res ##########################################
from functools import reduce

file_info = [
    ("RESULTS/qH_depths/qH_depth_2rots_nopenalty.csv", "Depth_1"),
    ("RESULTS/qH_depths/qH_depth_3rots_nopenalty.csv", "Depth_2"),
    ("RESULTS/qH_depths/qH_depth_4rots_nopenalty.csv", "Depth_3"),
    ("RESULTS/qH_depths/qH_depth_5rots_nopenalty.csv", "Depth_4"),
    ("RESULTS/qH_depths/qH_depth_6rots_nopenalty.csv", "Depth_5"),
    ("RESULTS/qH_depths/qH_depth_7rots_nopenalty.csv", "Depth_6"),
    ("RESULTS/qH_depths/qH_depth_2rots_penalties.csv", "Depth_1_pen"),
    ("RESULTS/qH_depths/qH_depth_3rots_penalties.csv", "Depth_2_pen"),
    ("RESULTS/qH_depths/qH_depth_4rots_penalties.csv", "Depth_3_pen"),
    ("RESULTS/qH_depths/qH_depth_5rots_penalties.csv", "Depth_4_pen"),
    ("RESULTS/qH_depths/qH_depth_6rots_penalties.csv", "Depth_5_pen"),
    ("RESULTS/qH_depths/qH_depth_7rots_penalties.csv", "Depth_6_pen")
]

# Load datasets
dfs = [pd.read_csv(file).rename(columns={"Depth": depth}) for file, depth in file_info]

# Merge on 'Size' column
df = reduce(lambda left, right: pd.merge(left, right, on="Size", how="outer"), dfs)
df.sort_values(by="Size", inplace=True)
df.interpolate(inplace=True)

# Plot configuration
plt.style.use("ggplot")
fig, ax1 = plt.subplots(figsize=(11, 10))

# Define color and marker styles
plot_settings = [
    ("Depth_1", 'b', 'o', '-'),
    ("Depth_2", 'r', 's', '--'),
    ("Depth_3", 'g', '^', '-.'),
    ("Depth_4", 'm', 'x', ':'),
    ("Depth_5", 'c', 'd', '--'),
    ("Depth_6", 'y', '*', '-.'),
]

# Define settings for penalty data (dashed, different markers)
penalty_settings = [
    ("Depth_1_pen", 'b', 's', ':'),
    ("Depth_2_pen", 'r', 's', ':'),
    ("Depth_3_pen", 'g', '^', '--'),
    ("Depth_4_pen", 'm', 'x', '-.'),
    ("Depth_5_pen", 'c', 'd', ':'),
    ("Depth_6_pen", 'y', '*', '--'),
]

label_map = {
    "Depth_1": "2 rot no penalties",
    "Depth_2": "3 rot no penalties",
    "Depth_3": "4 rot no penalties",
    "Depth_4": "5 rot no penalties",
    "Depth_5": "6 rot no penalties",
    "Depth_6": "7 rot no penalties",
    "Depth_1_pen": "2 rot penalties",
    "Depth_2_pen": "3 rot penalties",
    "Depth_3_pen": "4 rot penalties",
    "Depth_4_pen": "5 rot penalties",
    "Depth_5_pen": "6 rot penalties",
    "Depth_6_pen": "7 rot penalties",
}


# Plot depth with penalties
for depth, color, marker, linestyle in plot_settings:
    ax1.plot(
        df["Size"], df[depth],
        marker=marker,
        color=color,
        linestyle=linestyle,
        label=label_map[depth]
    )

for depth, color, marker, linestyle in penalty_settings:
    ax1.plot(
        df["Size"], df[depth],
        marker=marker,
        color=color,
        linestyle=linestyle,
        alpha=0.7,
        label=label_map[depth]
    )

ax1.set_xlabel("Number of Qubits", fontsize=24)
ax1.set_ylabel("Circuit Depth", fontsize=24)
ax1.xaxis.set_tick_params(labelsize=22)
ax1.yaxis.set_tick_params(labelsize=22)
ax1.legend(fontsize=22)
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# Save and display
pdf_filename = "Paper Plots/qH_circuit_depth_tot.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
plt.show()

# %% #################################  one plot per num rots ##########################################
from functools import reduce
import re
import numpy as np

# File list for all depths (qH and total, with and without penalties)
file_info = [
    ("RESULTS/qH_depths/total_depth_2rots_nopenalty.csv", "Total_1"),
    ("RESULTS/qH_depths/total_depth_2rots_penalties.csv", "Total_1_pen"),
    ("RESULTS/qH_depths/total_depth_2rots_baseline.csv", "Baseline_1"),
    
    ("RESULTS/qH_depths/total_depth_3rots_nopenalty.csv", "Total_2"),
    ("RESULTS/qH_depths/total_depth_3rots_penalties.csv", "Total_2_pen"),
    ("RESULTS/qH_depths/total_depth_3rots_baseline.csv", "Baseline_2"),

    ("RESULTS/qH_depths/total_depth_4rots_nopenalty.csv", "Total_3"),
    ("RESULTS/qH_depths/total_depth_4rots_penalties.csv", "Total_3_pen"),
    ("RESULTS/qH_depths/total_depth_4rots_baseline.csv", "Baseline_3"),
    
    ("RESULTS/qH_depths/total_depth_5rots_nopenalty.csv", "Total_4"),
    ("RESULTS/qH_depths/total_depth_5rots_penalties.csv", "Total_4_pen"),   
    ("RESULTS/qH_depths/total_depth_5rots_baseline.csv", "Baseline_4"),

    ("RESULTS/qH_depths/total_depth_6rots_nopenalty.csv", "Total_5"),
    ("RESULTS/qH_depths/total_depth_6rots_penalties.csv", "Total_5_pen"),
    ("RESULTS/qH_depths/total_depth_6rots_baseline.csv", "Baseline_5"),

    ("RESULTS/qH_depths/total_depth_7rots_nopenalty.csv", "Total_6"),
    ("RESULTS/qH_depths/total_depth_7rots_penalties.csv", "Total_6_pen"),
    ("RESULTS/qH_depths/total_depth_7rots_baseline.csv", "Baseline_6")
]

# Load and merge
dfs = [pd.read_csv(file).rename(columns={"Depth": depth}) for file, depth in file_info]
df = reduce(lambda left, right: pd.merge(left, right, on="Size", how="outer"), dfs)
df.sort_values(by="Size", inplace=True)
df.interpolate(inplace=True)

# Label map
label_map = {
    "Total": "Total Depth XY-QAOA",
    "Total_pen": "Total Depth QAOA",
    "Baseline": "Total Depth no penalties + X Mixer (Baseline)"
}

# Plot setup
fig, axes = plt.subplots(2, 3, figsize=(22, 14), sharex=True, sharey=True)
axes = axes.flatten()

# colors = ['#0072B2',  # Blue
#           '#009E73',  # Green
#           '#D55E00',  # Vermilion (orange-red)
#           '#CC79A7']  # Reddish purple
colors = ['royalblue',  # Blue
          'darkorange',  # Green
          'slategray']  # Baseline
markers = ['o', 's', 'x']
linestyles = ['--', '--', ':']


# Function to extract num_rot from filename using regex
def extract_num_rot(filename):
    match = re.search(r'(\d+)rots', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract num_rot from filename: {filename}")

# Loop over each file and its corresponding label
# manual_init_depths = {
#     2: 6,
#     3: 11,
#     4: 15,
#     5: 20,
#     6: 25,
#     7: 30
# }

# # Loop over each file and its corresponding label
# for file_path, column_name in file_info:
#     if column_name not in df.columns:
#         continue

#     num_rot = extract_num_rot(file_path)

#     # Only apply manual constant to Total_* (non-penalized)
#     if column_name.startswith("Total_") and "_pen" not in column_name:
#         if num_rot in manual_init_depths:
#             init_depth = manual_init_depths[num_rot]
#             df[column_name] = df[column_name] + init_depth
#             print(f"✅ Added constant init depth {init_depth} to {column_name}")
#         else:
#             print(f"⚠️ No manual init depth defined for num_rot = {num_rot}")

#     # Penalized Total_*: add +1 only
#     elif column_name.startswith("Total_") and "_pen" in column_name:
#         df[column_name] = df[column_name] + 1

for i in range(6):
    rot_num = i + 2  # Fix: start from 2, not 1
    ax = axes[i]

    curves = [
        (f"Total_{i+1}", label_map["Total"]),
        (f"Total_{i+1}_pen", label_map["Total_pen"]),
        (f"Baseline_{i+1}", label_map["Baseline"])
    ]

    for j, (col, label) in enumerate(curves):
        if col in df:
            ax.plot(df["Size"], df[col], color=colors[j], marker=markers[j], markersize=4, linestyle=linestyles[j], label=label)

    ax.set_title(f"{rot_num} Rotamers", fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)

axes[0].set_ylabel("Circuit Depth", fontsize=18)
axes[3].set_ylabel("Circuit Depth", fontsize=18)
for ax in axes[3:]:
    ax.set_xlabel("Number of Qubits", fontsize=18)

plt.tight_layout()
plt.savefig("Paper Plots/qH_depth_grid.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# File list for 4 rotamers
file_info = [
    ("RESULTS/qH_depths/total_depth_4rots_nopenalty.csv", "Total_3"),
    ("RESULTS/qH_depths/total_depth_4rots_penalties.csv", "Total_3_pen"),
    ("RESULTS/qH_depths/total_depth_4rots_baseline.csv", "Baseline_3")
]

# Load and merge data
dfs = [pd.read_csv(file).rename(columns={"Depth": depth}) for file, depth in file_info]
df = reduce(lambda left, right: pd.merge(left, right, on="Size", how="outer"), dfs)
df.sort_values(by="Size", inplace=True)
df.interpolate(inplace=True)

# Plot styles and labels
label_map = {
    "Total_3": "Total Depth XY-QAOA",
    "Total_3_pen": "Total Depth QAOA",
    "Baseline_3": "Total Depth no penalties + X Mixer (Baseline)"
}

# colors = ['#0072B2',  # Blue
#           '#009E73',  # Green
#           '#D55E00',  # Vermilion (orange-red)
#           '#CC79A7']  # Reddish purple
colors = ['royalblue',  # Blue
          'darkorange',  # Green
          'slategray']  # Baseline
markers = ['o', 's', 'x']
linestyles = ['--', '--', ':']

init_df = pd.read_csv("symmetry_init_depths.csv")

if "Total_3" in df.columns:
    df["Total_3"] = df["Total_3"] + 15  

if "Total_3_pen" in df.columns:
    df["Total_3_pen"] = df["Total_3_pen"] + 1

# Set up single plot
plt.figure(figsize=(10, 7))
for i, (col, label) in enumerate(label_map.items()):
    if col in df:
        plt.plot(df["Size"], df[col], label=label, color=colors[i],
                 marker=markers[i], markersize=5, linestyle=linestyles[i])

plt.xlabel("Number of Qubits", fontsize=18)
plt.ylabel("Circuit Depth", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save and show
plt.tight_layout()
plt.savefig("Paper Plots/qH_depth_4rots_state_prep.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from functools import reduce

# --- Setup ---
source_dir = "RESULTS/qH_depths"
rotamer_range = range(2, 8)  # Rotamers 2 to 7

# Collect file paths and labels
file_info = []
for i, rot in enumerate(rotamer_range, start=1):
    file_info.extend([
        (os.path.join(source_dir, f"total_cnots_{rot}rots_nopenalty.csv"), f"Total_{i}"),
        (os.path.join(source_dir, f"total_cnots_{rot}rots_penalties.csv"), f"Total_{i}_pen"),
        (os.path.join(source_dir, f"total_cnots_{rot}rots_baseline.csv"), f"Baseline_{i}"),
        (os.path.join(source_dir, f"total_cnots_{rot}rots_state_prep.csv"), f"Init_{i}")
    ])

# --- Load and merge all CSVs ---
dfs = [pd.read_csv(file).rename(columns={"CNOTs": label}) for file, label in file_info if os.path.exists(file)]
df = reduce(lambda left, right: pd.merge(left, right, on="Size", how="outer"), dfs)
df.sort_values(by="Size", inplace=True)
df.interpolate(inplace=True)

# --- Plot setup ---
def plot_cnot_grid(df, include_init_in_total=False, save_suffix=""):
    label_map = {
        "Total": "XY-QAOA (no penalty)",
        "Total_pen": "QAOA (with penalties)",
        "Baseline": "Baseline (X-mixer)"
    }

    colors = ['royalblue', 'darkorange', 'slategray']
    markers = ['o', 's', 'x']
    linestyles = ['--', '--', ':']

    fig, axes = plt.subplots(2, 3, figsize=(22, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(6):
        rot_num = i + 2
        ax = axes[i]

        # Prepare columns
        total_col = f"Total_{i+1}"
        init_col = f"Init_{i+1}"
        pen_col = f"Total_{i+1}_pen"
        base_col = f"Baseline_{i+1}"

        curves = [
            (total_col, label_map["Total"]),
            (pen_col, label_map["Total_pen"]),
            (base_col, label_map["Baseline"])
        ]

        # Modify total if requested
        if include_init_in_total and total_col in df and init_col in df:
            df[total_col] = df[total_col] + df[init_col]
            print(f"✅ Added Init_{i+1} to Total_{i+1}")

        for j, (col, label) in enumerate(curves):
            if col in df:
                ax.plot(df["Size"], df[col], color=colors[j], marker=markers[j], markersize=4, linestyle=linestyles[j], label=label)

        ax.set_title(f"{rot_num} Rotamers", fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=14)
        if i == 0:
            ax.legend(fontsize=14)

    axes[0].set_ylabel("CNOT Gate Count", fontsize=18)
    axes[3].set_ylabel("CNOT Gate Count", fontsize=18)
    for ax in axes[3:]:
        ax.set_xlabel("Number of Qubits", fontsize=18)

    plt.tight_layout()
    plt.savefig(f"Paper Plots/qH_CNOT_grid{save_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# --- Generate both plots ---
plot_cnot_grid(df.copy(), include_init_in_total=False, save_suffix="_no_init")
plot_cnot_grid(df.copy(), include_init_in_total=True, save_suffix="_with_init")

# %% ################################# plot compiled CNOTs ##########################################
import pandas as pd
import matplotlib.pyplot as plt

# Set the number of rotations (replace this with your desired value)
for rot in range(2,8):

    # Construct file paths
    file_path_depth = f"RESULTS/CNOTs/total_cnots_{rot}rots_XY.csv"
    file_name_baseline = f"RESULTS/CNOTs/total_cnots_{rot}rots_baseline.csv"
    file_name_penalty = f"RESULTS/CNOTs/total_cnots_{rot}rots_penalty.csv"

    # Load data
    df_xy = pd.read_csv(file_path_depth)
    df_baseline = pd.read_csv(file_name_baseline)
    df_penalty = pd.read_csv(file_name_penalty)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df_xy['Size'], df_xy['CNOTs'], marker='o', label='XY', color='royalblue')
    plt.plot(df_baseline['Size'], df_baseline['CNOTs'], marker='s', label='Baseline', color='slategray')
    plt.plot(df_penalty['Size'], df_penalty['CNOTs'], marker='^', label='Penalty', color='darkorange')

    plt.xlabel('Size')
    plt.ylabel('Number of CNOT Gates with Initial State Prep.')
    plt.title(f'CNOTs vs Size for {rot} Rotamers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# %% ################################# grid plot compiled CNOTs with initial prep. ##########################################
import pandas as pd
import matplotlib.pyplot as plt

# Define range of rotamers and layout
rotamer_range = range(2, 8)
n_cols = 3
n_rows = (len(rotamer_range) + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten to easily index subplots

for idx, rot in enumerate(rotamer_range):
    ax = axes[idx]

    file_path_depth = f"RESULTS/CNOTs/total_cnots_{rot}rots_XY.csv"
    file_name_baseline = f"RESULTS/CNOTs/total_cnots_{rot}rots_baseline.csv"
    file_name_penalty = f"RESULTS/CNOTs/total_cnots_{rot}rots_penalty.csv"

    # Load CSVs
    df_xy = pd.read_csv(file_path_depth)
    df_baseline = pd.read_csv(file_name_baseline)
    df_penalty = pd.read_csv(file_name_penalty)

    # Plot on subplot
    ax.plot(df_xy['Size'], df_xy['CNOTs'], marker='o', label='XY', color='royalblue')
    ax.plot(df_baseline['Size'], df_baseline['CNOTs'], marker='s', label='Baseline', color='slategray')
    ax.plot(df_penalty['Size'], df_penalty['CNOTs'], marker='^', label='Penalty', color='darkorange')

    ax.set_title(f'{rot} Rotamers')
    ax.grid(True)

    if idx % n_cols == 0:
        ax.set_ylabel('CNOT Gates')
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Size')

# Hide any unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

# Add legend only once
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03))
fig.suptitle('Transpiled CNOTs With State Prep.', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.90)  # make space for legend
plt.show()


# %% ################################# plot compiled CNOTs no initial state ##########################################
import pandas as pd
import matplotlib.pyplot as plt

# Set the number of rotations (replace this with your desired value)
for rot in range(2,8):

    # Construct file paths
    file_path_depth = f"RESULTS/CNOTs/total_cnots_{rot}rots_XY_no_init.csv"
    file_name_baseline = f"RESULTS/CNOTs/total_cnots_{rot}rots_baseline_no_init.csv"
    file_name_penalty = f"RESULTS/CNOTs/total_cnots_{rot}rots_penalty_no_init.csv"

    # Load data
    df_xy = pd.read_csv(file_path_depth)
    df_baseline = pd.read_csv(file_name_baseline)
    df_penalty = pd.read_csv(file_name_penalty)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df_xy['Size'], df_xy['CNOTs'], marker='o', label='XY', color='royalblue')
    plt.plot(df_baseline['Size'], df_baseline['CNOTs'], marker='s', label='Baseline', color='slategray')
    plt.plot(df_penalty['Size'], df_penalty['CNOTs'], marker='^', label='Penalty', color='darkorange')

    plt.xlabel('Size')
    plt.ylabel('Number of CNOT Gates')
    plt.title(f'CNOTs vs Size for {rot} Rotamers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% ################################# grid plot compiled CNOTs without initial prep. ##########################################
import pandas as pd
import matplotlib.pyplot as plt

# Define range of rotamers and subplot grid size
rotamer_range = range(2, 8)
n_cols = 3
n_rows = (len(rotamer_range) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.flatten()

for idx, rot in enumerate(rotamer_range):
    ax = axes[idx]

    file_path_depth = f"RESULTS/CNOTs/total_cnots_{rot}rots_XY_no_init.csv"
    file_name_baseline = f"RESULTS/CNOTs/total_cnots_{rot}rots_baseline_no_init.csv"
    file_name_penalty = f"RESULTS/CNOTs/total_cnots_{rot}rots_penalty_no_init.csv"

    df_xy = pd.read_csv(file_path_depth)
    df_baseline = pd.read_csv(file_name_baseline)
    df_penalty = pd.read_csv(file_name_penalty)

    ax.plot(df_xy['Size'], df_xy['CNOTs'], marker='o', label='XY', color='royalblue')
    ax.plot(df_baseline['Size'], df_baseline['CNOTs'], marker='s', label='Baseline', color='slategray')
    ax.plot(df_penalty['Size'], df_penalty['CNOTs'], marker='^', label='Penalty', color='darkorange')

    ax.set_title(f'{rot} Rotamers')
    ax.grid(True)

    if idx % n_cols == 0:
        ax.set_ylabel('CNOT Gates')
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Size')

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

# Single legend on top
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

fig.suptitle('Transpiled CNOTs Without State Prep.', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# %%
