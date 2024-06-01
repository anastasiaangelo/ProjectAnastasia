import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import ast

# # Define the paths
# folder_paths = {
#     "XY": "RESULTS/XY-QAOA",
#     "XY_3rot": "RESULTS/3rot-XY-QAOA",
#     "no_penalty": "RESULTS/nopenalty-QAOA"
# }

file_XY_3rot = "RESULTS/3rot-XY-QAOA/2res-3rot.csv"

# def custom_sort_key(filename):
#     numbers = re.findall(r'\d+', filename)
#     return [int(num) for num in numbers]

# # Function to safely read CSV files with custom sorting
# def read_sorted_csv(folder):
#     files = sorted(os.listdir(folder), key=custom_sort_key)  # Sort with the custom key
#     data_frames = []
#     for f in files:
#         if f.endswith(".csv"):
#             try:
#                 df = pd.read_csv(os.path.join(folder, f), on_bad_lines='skip')
#                 data_frames.append(df)
#             except pd.errors.ParserError as e:
#                 print(f"Error reading {f}: {e}")
#     return data_frames

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating string: {value}, {e}")
        return None

# Read data
# data_XY = read_sorted_csv(folder_paths["XY"])
# data_3rot_XY = read_sorted_csv(folder_paths["XY_3rot"])
# data_no_penalty = read_sorted_csv(folder_paths["no_penalty"])

df_XY3 = pd.read_csv(file_XY_3rot, on_bad_lines='skip').tail(1)

# Store ratios and num_qubits
probabilities_2 = []
sorted_bitstrings_2= []

probabilities_3 = []
sorted_bitstrings_3 = []

# Calculate the ratios
# for df_XY in data_XY:
#     sorted_bitstrings_str = df_XY['Sorted Bitstrings'].values[-1]
#     sorted_bitstrings = ast.literal_eval(sorted_bitstrings_str)
#     frac_XY = df_XY['Fraction'].values[-1]
    
#     fractions.append(frac_XY)
#     sorted_bitstrings_3.append([bitstring for bitstring, data in sorted_bitstrings])

# for df_XY_3rot in data_3rot_XY:
#     sorted_bitstrings_str = df_XY_3rot['Sorted Bitstrings'].values[-1]
#     sorted_bitstrings = safe_literal_eval(sorted_bitstrings_str)
#     if sorted_bitstrings is not None:
#         probabilities_3.append([data['probability'] for bitstring, data in sorted_bitstrings])
#         sorted_bitstrings_3.append([bitstring for bitstring, data in sorted_bitstrings])

if not df_XY3['Sorted Bitstrings'].isnull().values.any():
    sorted_bitstrings_str = df_XY3['Sorted Bitstrings'].values[-1]
    shots = df_XY3['shots']
    total_bitstrings = df_XY3['Total Bitstrings']
    sorted_bitstrings = safe_literal_eval(sorted_bitstrings_str)
    if sorted_bitstrings is not None:
        probabilities_3 = [data['probability'] for bitstring, data in sorted_bitstrings]
        sorted_bitstrings_3 = [bitstring for bitstring, data in sorted_bitstrings]


print(sorted_bitstrings_2)
print(sorted_bitstrings_3)
print(probabilities_2)
print(probabilities_3)

print(range(len(sorted_bitstrings_3)))
print(range(len(probabilities_3)))

# flattened_bitstrings_2 = [bitstring for sublist in sorted_bitstrings_2 for bitstring in sublist]
# flattened_bitstrings_3 = [bitstring for sublist in sorted_bitstrings_3 for bitstring in sublist]

# flattened_probabilities_2 = [prob for sublist in probabilities_2 for prob in sublist]
# flattened_probabilities_3 = [prob for sublist in probabilities_3 for prob in sublist]

flattened_bitstrings_2 = sorted_bitstrings_2
flattened_bitstrings_3 = sorted_bitstrings_3

flattened_probabilities_2 = probabilities_2
flattened_probabilities_3 = probabilities_3

# Plotting the ratios
fig, ax = plt.subplots()
x = range(len(flattened_bitstrings_2))
x3 = range(len(flattened_bitstrings_3))

# ax.plot(x, y_XY, label="XY mixer 2rot", marker='x')
ax.plot(x3, flattened_probabilities_3, label="XY mixer 3 rot", marker='1')

ax.set_xlabel("Bitstrings")
ax.set_ylabel("Probability")
ax.set_title("Probability distributions of good bitstrings by Sorted Bitstrings")
ax.legend()

ax.set_xticks(x3)

plt.tight_layout()
plt.savefig("Paper Plots/prob_distributions.pdf")
plt.show()
