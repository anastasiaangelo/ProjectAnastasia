# %%  import
import pandas as pd
import matplotlib.pyplot as plt

# %% ################################# One curve per num rots per res ##########################################
# Load datasets
depth2_df = pd.read_csv("RESULTS/qH_depths/qH_depth_2rots.csv").rename(columns={"Depth": "Depth_2rots"})
depth3_df = pd.read_csv("RESULTS/qH_depths/qH_depth_3rots.csv").rename(columns={"Depth": "Depth_3rots"})
depth4_df = pd.read_csv("RESULTS/qH_depths/qH_depth_4rots.csv").rename(columns={"Depth": "Depth_4rots"})
depth5_df = pd.read_csv("RESULTS/qH_depths/qH_depth_5rots.csv").rename(columns={"Depth": "Depth_5rots"})
depth6_df = pd.read_csv("RESULTS/qH_depths/qH_depth_6rots.csv").rename(columns={"Depth": "Depth_6rots"})

# Merge datasets on 'Size'
df = depth2_df
for depth_df in [depth3_df, depth4_df, depth5_df, depth6_df]:
    df = pd.merge(df, depth_df, on="Size", how="outer")

# Sort by Circuit Size
df = df.sort_values(by="Size")
df.interpolate(inplace=True)  # Fill missing values smoothly

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

# Plot depths for different rotamer residues
ax.plot(df["Size"], df["Depth_2rots"], marker='o', linestyle='-', label="Depth 2 rots", color='b')
ax.plot(df["Size"], df["Depth_3rots"], marker='s', linestyle='--', label="Depth 3 rots", color='r')
ax.plot(df["Size"], df["Depth_4rots"], marker='^', linestyle='-.', label="Depth 4 rots", color='g')
ax.plot(df["Size"], df["Depth_5rots"], marker='x', linestyle='--', label="Depth 5 rots", color='c')
ax.plot(df["Size"], df["Depth_6rots"], marker='d', linestyle='-', label="Depth 6 rots", color='m')

# Labels and title
ax.set_xlabel("Circuit Size")
ax.set_ylabel("Circuit Depth")
ax.set_title("Circuit Depth per number of Rotamers vs. Size")
ax.legend()
ax.grid(True)

# Save the plot
pdf_filename = "Paper Plots/qH_circuit_depth_nrots.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")  # Ensures no clipping

plt.show()


# %% ################################# One curve for all values ##########################################
# Load dataset
df = pd.read_csv("RESULTS/qH_depths/qH_depths.csv")

# Sort by Circuit Size
df = df.sort_values(by="Size")
df.interpolate(inplace=True)  # Smooth missing values if necessary

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

# Plot Depth as a line with markers for better clarity
ax.plot(df["Size"], df["Depth"], marker='o', linestyle='-', color='b', label="Circuit Depth", alpha=0.8)

# Optional: Add a scatter plot on top for better visualization
ax.scatter(df["Size"], df["Depth"], color='b', edgecolors='black', zorder=3, label="Data Points")

# Labels and title
ax.set_xlabel("Circuit Size")
ax.set_ylabel("Circuit Depth")
ax.set_title("Circuit Depth vs. Size")

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Save the improved plot
pdf_filename = "Paper Plots/qH_circuit_depth.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")  # Ensures no clipping

plt.show()
# %% ################################# Penalties vs. no penalties ##########################################
df1 = pd.read_csv("RESULTS/qH_depths/qH_depths_penalties.csv").rename(columns={"Depth": "Depth_with_penalty"})
df2 = pd.read_csv("RESULTS/qH_depths/qH_depths_nopenalty.csv").rename(columns={"Depth": "Depth_no_penalty"})

df = pd.merge(df1, df2, on="Size", how="outer").sort_values(by="Size")
df.interpolate(inplace=True)  # Fill missing values smoothly

fig, ax = plt.subplots(figsize=(9, 6))

# Plot Depth with penalties
ax.plot(df["Size"], df["Depth_with_penalty"], marker='o', linestyle='-', color='b', alpha=0.8, label="With Penalty")
ax.scatter(df["Size"], df["Depth_with_penalty"], color='b', edgecolors='black', zorder=3)

# Plot Depth without penalties
ax.plot(df["Size"], df["Depth_no_penalty"], marker='^', linestyle='--', color='g', alpha=0.8, label="No Penalty")
ax.scatter(df["Size"], df["Depth_no_penalty"], color='g', edgecolors='black', zorder=3)

ax.set_xlabel("Circuit Size")
ax.set_ylabel("Circuit Depth")
ax.set_title("Circuit Depth: Penalties vs No Penalties")

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

pdf_filename = "Paper Plots/qH_circuit_depth_penalties_vs_nopenalties.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight") 

plt.show()


# %% #################################  num rots per res ##########################################
from functools import reduce

file_info = [
    ("RESULTS/qH_depths/qH_depth_2rots.csv", "Depth_1"),
    ("RESULTS/qH_depths/qH_depth_3rots_nopenalty.csv", "Depth_2"),
    ("RESULTS/qH_depths/qH_depth_4rots_nopenalty.csv", "Depth_3"),
    ("RESULTS/qH_depths/qH_depth_5rots_nopenalty.csv", "Depth_4"),
    ("RESULTS/qH_depths/qH_depth_6rots_nopenalty.csv", "Depth_5"),
    ("RESULTS/qH_depths/qH_depth_7rots_nopenalty.csv", "Depth_6"),
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
fig, ax1 = plt.subplots(figsize=(9, 5))

# Define color and marker styles
plot_settings = [
    ("Depth_1", 'b', 'o', '-'),
    ("Depth_2", 'r', 's', '--'),
    ("Depth_3", 'g', '^', '-.'),
    ("Depth_4", 'm', 'x', ':'),
    ("Depth_5", 'c', 'd', '--'),
    ("Depth_6", 'y', '*', '-.'),
]

# Plot depth without penalties
for depth, color, marker, linestyle in plot_settings:
    ax1.plot(df["Size"], df[depth], marker=marker, color=color, linestyle=linestyle, label=f"{depth.replace('_', ' ')}")

# Define settings for penalty data (dashed, different markers)
penalty_settings = [
    ("Depth_2_pen", 'r', 's', ':'),
    ("Depth_3_pen", 'g', '^', '--'),
    ("Depth_4_pen", 'm', 'x', '-.'),
    ("Depth_5_pen", 'c', 'd', ':'),
    ("Depth_6_pen", 'y', '*', '--'),
]

# Plot depth with penalties
for depth, color, marker, linestyle in penalty_settings:
    ax1.plot(df["Size"], df[depth], marker=marker, color=color, linestyle=linestyle, alpha=0.7, label=f"{depth.replace('_', ' ')} (penalty)")

ax1.set_xlabel("Circuit Size")
ax1.set_ylabel("Circuit Depth")
ax1.set_title("Circuit Depth vs. Circuit Size")

# Legend & grid
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Save and display
pdf_filename = "Paper Plots/qH_circuit_depth_tot.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
plt.show()

# %%
