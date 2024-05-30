import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
folder_paths = {
    "local": "RESULTS/localpenalty-QAOA",
    "XY": "RESULTS/XY-QAOA",
    "no_penalty": "RESULTS/nopenalty-QAOA",
    "hardware": "RESULTS/sessionid"
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
    print(f"Read {len(data_frames)} files from {folder}")

    return data_frames

# Read data
data_local = read_sorted_csv(folder_paths["local"])
data_XY = read_sorted_csv(folder_paths["XY"])
data_no_penalty = read_sorted_csv(folder_paths["no_penalty"])
data_hardware = read_sorted_csv(folder_paths["hardware"])

# Determine the maximum number of files across all folders
max_files = max(len(data_local), len(data_XY), len(data_no_penalty), len(data_hardware))
print(f"Maximum number of files across all folders: {max_files}")

# Store ratios and num_qubits
ratios = []
num_qubits_list = []
num_qubits_hw_dict = {}

# Calculate the ratios
for i in range(max_files):
    energy_local = energy_XY = energy_classical = energy_hardware = None
    num_qubits = num_qubits_hw = None

    if i < len(data_local):
        df_local = data_local[i]
        energy_local = df_local['Ground State Energy'].values[-1]

    if i < len(data_XY):
        df_XY = data_XY[i]
        energy_XY = df_XY['Ground State Energy'].values[-1]

    if i < len(data_no_penalty):
        df_no_penalty = data_no_penalty[i]
        energy_classical = df_no_penalty['Ground State Energy'].values[-1]
        num_qubits = df_no_penalty['Number of qubits'].values[-1]

    if i < len(data_hardware):
        df_hardware = data_hardware[i]
        energy_hardware = df_hardware['Ground State Energy'].values[-1]
        num_qubits_hw = df_hardware['Number of qubits'].values[-1]

    if energy_classical is not None:
        ratio_XY_classical = energy_XY / energy_classical if energy_XY is not None else None
        ratio_local_classical = energy_local / energy_classical if energy_local is not None else None

        ratios.append((ratio_XY_classical, ratio_local_classical))
        num_qubits_list.append(num_qubits)

        if energy_hardware is not None:
            ratio_hw_classical = energy_hardware / energy_classical
            if num_qubits_hw not in num_qubits_hw_dict:
                num_qubits_hw_dict[num_qubits_hw] = ratio_hw_classical


# Debugging statements to verify lengths
print(f"num_qubits_list: {num_qubits_list}")
print(f"ratios: {ratios}")
print(f"num_qubits_hw_list: {num_qubits_hw_dict}")

# Filter out None values for plotting
filtered_ratios = [r for r in ratios if None not in r]
filtered_num_qubits_list = [q for q, r in zip(num_qubits_list, ratios) if None not in r]

# Extract ratios for plotting
num_qubits_hw = sorted(num_qubits_hw_dict.keys())
y_hw_classical = [num_qubits_hw_dict[q] for q in num_qubits_hw]

# Plotting the ratios
fig, ax = plt.subplots()
x = num_qubits_list
x_hw = num_qubits_hw
y_XY_classical = [r[0] for r in ratios]
y_local_classical = [r[1] for r in ratios]

ax.plot(x, y_XY_classical, label="XY/Classical", marker='o')
ax.plot(x, y_local_classical, label="Local/Classical", marker='x')
ax.plot(x_hw, y_hw_classical, label="Hardware/Classical", marker='^')

ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Energy Ratio")
ax.set_title("Energy Ratios by Number of Qubits")
ax.legend()
plt.savefig('Energy_2rot.pdf')
plt.show()
