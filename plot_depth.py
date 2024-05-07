import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
folder_paths = {
    "local": "RESULTS/Depths/localpenalty-QAOA",
    "XY": "RESULTS/Depths/XY-QAOA",
    "no_penalty": "RESULTS/nopenalty-QAOA"
}

def custom_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

# Function to safely read CSV files with custom sorting
def read_sorted_csv(folder):
    files = sorted(os.listdir(folder), key=custom_sort_key)  # Sort with the custom key
    return [pd.read_csv(os.path.join(folder, f)) for f in files if f.endswith(".csv")]

# Read data
data_local = read_sorted_csv(folder_paths["local"])
data_XY = read_sorted_csv(folder_paths["XY"])
data_no_penalty = read_sorted_csv(folder_paths["no_penalty"])

# Store ratios and num_qubits
depths = []
num_qubits_list = []

# Calculate the ratios
for df_local, df_XY, df_no_penalty in zip(data_local, data_XY, data_no_penalty):
    num_qubits = df_no_penalty['Number of qubits']  # From no_penalty file
    depth_local = df_local['Depth of the circuit']
    depth_XY = df_XY['Depth of the circuit']
    
    depths.append((depth_local, depth_XY))
    num_qubits_list.append(num_qubits)

num_qubits_list = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
print(num_qubits_list)
print(depths)

# Plotting the ratios
fig, ax = plt.subplots()
x = num_qubits_list
y_local = [r[0] for r in depths]
y_XY = [r[1] for r in depths]

ax.plot(x, y_local, label="local penalties", marker='o')
ax.plot(x, y_XY, label="XY mixer", marker='x')

ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Depth of Circuit")
ax.set_title("Circuit Depth by Number of Qubits")
ax.legend()
plt.show()
