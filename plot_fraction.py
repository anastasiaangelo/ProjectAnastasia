import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
folder_paths = {
    "local": "RESULTS/localpenalty-QAOA",
    "XY": "RESULTS/XY-QAOA",
    "no_penalty": "RESULTS/nopenalty-QAOA",
    "local_3rot": "RESULTS/3rot-localpenalty-QAOA",
    "XY_3rot": "RESULTS/3rot-XY-QAOA"
}

def custom_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

# Function to safely read CSV files with custom sorting
def read_sorted_csv(folder):
    files = sorted(os.listdir(folder), key=custom_sort_key)  # Sort with the custom key
    data_frames = []
    for f in files:
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder, f), on_bad_lines='skip')
                data_frames.append(df)
            except pd.errors.ParserError as e:
                print(f"Error reading {f}: {e}")
    return data_frames

# Read data
data_local = read_sorted_csv(folder_paths["local"])
data_XY = read_sorted_csv(folder_paths["XY"])
data_no_penalty = read_sorted_csv(folder_paths["no_penalty"])
data_3rot_local = read_sorted_csv(folder_paths["local_3rot"])
data_3rot_XY = read_sorted_csv(folder_paths["XY_3rot"])

# Store ratios and num_qubits
fractions = []
num_qubits_list = []

fractions_3 = []
num_qubits_3 = []

# Calculate the ratios
for df_local, df_XY, df_no_penalty in zip(data_local, data_XY, data_no_penalty):
    num_qubits = df_no_penalty['Number of qubits']  # From no_penalty file
    frac_local = df_local['Fraction'].values[-1]
    frac_XY = df_XY['Fraction'].values[-1]
    
    fractions.append((frac_local, frac_XY))
    num_qubits_list.append(num_qubits)

for df_local_3rot, df_XY_3rot in zip(data_3rot_local, data_3rot_XY):
    frac_local_3 = df_local_3rot['Fraction'].values[-1]
    frac_XY_3 = df_XY_3rot['Fraction'].values[-1]
    
    fractions_3.append((frac_local_3, frac_XY_3))

num_qubits_3 = [12, 15, 18, 21, 24]
print(num_qubits_list)
print(num_qubits_3)
print(fractions)

# Plotting the ratios
fig, ax = plt.subplots()
x = num_qubits_list
x3 = num_qubits_3
y_local = [r[0] for r in fractions]
y_XY = [r[1] for r in fractions]
y_local_3rot = [r[0] for r in fractions_3]
y_XY_3rot = [r[1] for r in fractions_3]

ax.plot(x, y_local, label="local penalties 2rot", marker='o')
ax.plot(x, y_XY, label="XY mixer 2rot", marker='x')
ax.plot(x3, y_local_3rot, label="local penalties 3rot", marker='^')
ax.plot(x3, y_XY_3rot, label="XY mixer 3 rot", marker='1')

ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Fractions")
ax.set_title("Fraction of good bitstrings by Number of Qubits")
ax.legend()
plt.savefig("Paper Plots/Fraction.pdf")
plt.show()
