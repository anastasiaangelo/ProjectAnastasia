import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
folder_paths = {
    "local": "RESULTS/localpenalty-QAOA",
    "XY": "RESULTS/XY-QAOA",
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
ratios = []
num_qubits_list = []

# Calculate the ratios
for df_local, df_XY, df_no_penalty in zip(data_local, data_XY, data_no_penalty):
    num_qubits = df_no_penalty['Number of qubits']  # From no_penalty file
    energy_local = df_local['Ground State Energy']
    energy_XY = df_XY['Ground State Energy']
    energy_classical = df_no_penalty['Ground State Energy']

    ratio_XY_classical = energy_XY / energy_classical
    ratio_local_classical = energy_local / energy_classical
    
    ratios.append((ratio_XY_classical, ratio_local_classical))
    num_qubits_list.append(num_qubits)


print(num_qubits_list)
print(ratios)

# Plotting the ratios
fig, ax = plt.subplots()
x = num_qubits_list
y_XY_classical = [r[0] for r in ratios]
y_local_classical = [r[1] for r in ratios]

ax.plot(x, y_XY_classical, label="XY/Classical", marker='o')
ax.plot(x, y_local_classical, label="Local/Classical", marker='x')

ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Energy Ratio")
ax.set_title("Energy Ratios by Number of Qubits")
ax.legend()
plt.show()
