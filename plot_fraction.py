import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
folder_paths = {
    "local": "RESULTS/2rot-localpenalty-QAOA",
    "XY": "RESULTS/2rot-XY-QAOA",
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
data_3rot_local = read_sorted_csv(folder_paths["local_3rot"])
data_3rot_XY = read_sorted_csv(folder_paths["XY_3rot"])

# Build dictionaries indexed by number of qubits
def build_qubit_dict(data_frames):
    out = {}
    for df in data_frames:
        if 'Number of qubits' in df.columns and 'Fraction' in df.columns:
            nq = int(df['Number of qubits'].values[0])
            out[nq] = df['Fraction'].values[-1]
    return out

dict_local = build_qubit_dict(data_local)
dict_XY = build_qubit_dict(data_XY)
dict_local_3 = build_qubit_dict(data_3rot_local)
dict_XY_3 = build_qubit_dict(data_3rot_XY)

# Collect all unique qubit counts
qubits_2rot = sorted(set(dict_local.keys()).union(dict_XY.keys()))
qubits_3rot = sorted(set(dict_local_3.keys()).union(dict_XY_3.keys()))

# Store fractions
fractions = [(dict_local.get(q), dict_XY.get(q)) for q in qubits_2rot]
fractions_3 = [(dict_local_3.get(q), dict_XY_3.get(q)) for q in qubits_3rot]

# Prepare data for plotting, filtering Nones
x_local = [q for q, (f_loc, _) in zip(qubits_2rot, fractions) if f_loc is not None]
y_local = [f_loc for f_loc, _ in fractions if f_loc is not None]

x_XY = [q for q, (_, f_xy) in zip(qubits_2rot, fractions) if f_xy is not None]
y_XY = [f_xy for _, f_xy in fractions if f_xy is not None]

x3_local = [q for q, (f_loc, _) in zip(qubits_3rot, fractions_3) if f_loc is not None]
y_local_3rot = [f_loc for f_loc, _ in fractions_3 if f_loc is not None]

x3_XY = [q for q, (_, f_xy) in zip(qubits_3rot, fractions_3) if f_xy is not None]
y_XY_3rot = [f_xy for _, f_xy in fractions_3 if f_xy is not None]

# Print summary
print("🧪 2-rotamers (local penalties vs XY):")
for q, (f_loc, f_xy) in zip(qubits_2rot, fractions):
    f_loc_str = f"{f_loc:.6f}" if f_loc is not None else "N/A"
    f_xy_str = f"{f_xy:.6f}" if f_xy is not None else "N/A"
    print(f"Qubits: {q:<3} | Local: {f_loc_str} | XY: {f_xy_str}")

print("\n🧪 3-rotamers (local penalties vs XY):")
for q, (f_loc, f_xy) in zip(qubits_3rot, fractions_3):
    f_loc_str = f"{f_loc:.6f}" if f_loc is not None else "N/A"
    f_xy_str = f"{f_xy:.6f}" if f_xy is not None else "N/A"
    print(f"Qubits: {q:<3} | Local: {f_loc_str} | XY: {f_xy_str}")

# Plotting

plt.figure(figsize=(11, 10))

plt.plot(x_local, y_local, label="local penalties 2rot", marker='o', color='darkorange')
plt.plot(x_XY, y_XY, label="XY mixer 2rot", marker='s', color='royalblue')
plt.plot(x3_local, y_local_3rot, label="local penalties 3rot", marker='o', color='red')
plt.plot(x3_XY, y_XY_3rot, label="XY mixer 3rot", marker='s', color='mediumseagreen')

plt.xlabel("Number of Qubits", fontsize=24)
plt.ylabel("Fraction of good bitstrings", fontsize=24)
plt.title("Fractions vs Number of Qubits")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=18, loc='lower right')
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("Paper Plots/Fraction.pdf")
plt.show()
