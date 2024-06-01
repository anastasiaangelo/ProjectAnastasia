import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths
# folder_paths = {
#     "local": "RESULTS/Depths/localpenalty-QAOA-noopt",
#     "XY": "RESULTS/Depths/XY-QAOA-noopt",
#     "no_penalty": "RESULTS/nopenalty-QAOA",
#     "local_3rot": "RESULTS/Depths/3rot-localpenalty-QAOA-noopt",
#     "XY_3rot": "RESULTS/Depths/3rot-XY-QAOA-noopt"
# }

# folder_paths = {
#     "local": "RESULTS/Depths/localpenalty-QAOA-basic",
#     "XY": "RESULTS/Depths/XY-QAOA-basic",
#     "no_penalty": "RESULTS/nopenalty-QAOA",
#     "local_3rot": "RESULTS/Depths/3rot-localpenalty-QAOA-basic",
#     "XY_3rot": "RESULTS/Depths/3rot-XY-QAOA-basic"
# }

folder_paths = {
    "local": "RESULTS/Depths/localpenalty-QAOA",
    "XY": "RESULTS/Depths/XY-QAOA",
    "no_penalty": "RESULTS/nopenalty-QAOA",
    "local_3rot": "RESULTS/Depths/3rot-localpenalty-QAOA",
    "XY_3rot": "RESULTS/Depths/3rot-XY-QAOA"
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
data_3rot_local = read_sorted_csv(folder_paths["local_3rot"])
data_3rot_XY = read_sorted_csv(folder_paths["XY_3rot"])

# Store ratios and num_qubits
depths = []
num_qubits_list = []

depths_3 = []
num_qubits_3 = []

# Calculate the ratios
for df_local, df_XY, df_no_penalty in zip(data_local, data_XY, data_no_penalty):
    num_qubits = df_no_penalty['Number of qubits']  # From no_penalty file
    depth_local = df_local['CNOTs']
    depth_XY = df_XY['CNOTs']
    
    depths.append((depth_local, depth_XY))
    num_qubits_list.append(num_qubits)

for df_local_3rot, df_XY_3rot in zip(data_3rot_local, data_3rot_XY):
    depth_local_3rot = df_local_3rot['CNOTs']
    depth_XY_3rot = df_XY_3rot['CNOTs']
    
    depths_3.append((depth_local_3rot, depth_XY_3rot))

num_qubits_3 = [12, 15, 18, 21, 24, 27, 30]
print(num_qubits_list)
print(num_qubits_3)
print(depths)

# Plotting the ratios
fig, ax = plt.subplots()
x = num_qubits_list
x3 = num_qubits_3
y_local = [r[0] for r in depths]
y_XY = [r[1] for r in depths]
y_local_3rot = [r[0] for r in depths_3]
y_XY_3rot = [r[1] for r in depths_3]

ax.plot(x, y_local, label="local penalties 2rot", marker='o')
ax.plot(x, y_XY, label="XY mixer 2rot", marker='x')
ax.plot(x3, y_local_3rot, label="local penalties 3rot", marker='^')
ax.plot(x3, y_XY_3rot, label="XY mixer 3 rot", marker='1')

ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Number of CNOTs")
ax.set_title("Number of CNOTs by Number of Qubits Stochastic Routing and 3 level optimisation")
ax.legend()
plt.savefig('Paper Plots/CNOTS_stochastic_3opt.pdf')
plt.show()
