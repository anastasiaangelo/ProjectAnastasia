import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
depth2_df = pd.read_csv("RESULTS/qH_depth_2rots.csv").rename(columns={"Depth": "Depth_1"})
depth3_df = pd.read_csv("RESULTS/qH_depth_3rots.csv").rename(columns={"Depth": "Depth_2"})
                                                               
# Merge on 'Size' column to align data points
df = pd.merge(depth2_df, depth3_df, on="Size", how="outer")

# Sort by Size to ensure correct plotting order
df = df.sort_values(by="Size")

df.interpolate(inplace=True)

# Create figure and first axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Depth on the first y-axis
ax1.plot(df["Size"], df["Depth_1"], marker='o', color='b', linestyle='-', label="Depth")
ax1.set_xlabel("Circuit Size")
ax1.set_ylabel("Circuit Depth 2 rots per res", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create secondary y-axis
ax2 = ax1.twinx()

# Plot Depth on the second y-axis
ax2.plot(df["Size"], df["Depth_2"], marker='s', color='r', linestyle='--', label="Depth 3 rots")
ax2.set_ylabel("Circuit Depth 3 rots", color ='r')
ax2.tick_params(axis='y', labelcolor='r')

# Title and grid
plt.title(f"Circuit Depth vs. Size")
ax1.grid(True)

# Show the plot


# Save the plot
pdf_filename = "Paper Plots/qH_circuit_depth.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")  # Ensures no clipping
plt.show()