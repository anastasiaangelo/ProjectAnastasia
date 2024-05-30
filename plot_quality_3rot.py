import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
folder_paths = {
    "local": "RESULTS/3rot-localpenalty-QAOA",
    "XY": "RESULTS/3rot-XY-QAOA",
    "no_penalty": "RESULTS/3rot_nopenalty-QAOA",
    "hardware": "RESULTS/hardware"
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
data_hardware = read_sorted_csv(folder_paths["hardware"])

# Store ratios and num_qubits
ratios = []
num_qubits_list = []
num_qubits_hw = []

# Calculate the ratios
for df_local, df_XY, df_no_penalty, df_hardware in zip(data_local, data_XY, data_no_penalty, data_hardware):
    num_qubits = df_no_penalty['Number of qubits'].values[-1] 

    energy_local = df_local['Ground State Energy'].values[-1]
    energy_XY = df_XY['Ground State Energy'].values[-1]
    energy_classical = df_no_penalty['Ground State Energy'].values[-1]
    energy_hardware = df_hardware['Ground State Energy']

    ratio_XY_classical = energy_XY / energy_classical
    ratio_local_classical = energy_local / energy_classical
    ratio_hw_classical = energy_hardware / energy_classical
    
    ratios.append((ratio_XY_classical, ratio_local_classical, ratio_hw_classical))
    num_qubits_list.append(num_qubits)

num_qubits_hw = [21, 24, 27, 30, 39]
print(num_qubits_list)
print(ratios)


# Plotting the ratios
fig, ax = plt.subplots()
x = num_qubits_list
x_hw = num_qubits_hw
y_XY_classical = [r[0] for r in ratios]
y_local_classical = [r[1] for r in ratios]
y_hw_classical = [r[2] for r in ratios]

ax.plot(x, y_XY_classical, label="XY/Classical", marker='o')
ax.plot(x, y_local_classical, label="Local/Classical", marker='x')
ax.plot(x_hw, y_hw_classical, label="Hardware/Classical", marker='^')

ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Energy Ratio")
ax.set_title("Energy Ratios by Number of Qubits for 3 rotamers per residue")
ax.legend()
plt.savefig('Energy_3rot.pdf')
plt.show()
