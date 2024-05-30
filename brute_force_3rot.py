# Point 1 of constraint studies for paper, Ising model with no penalties, constriants enforced by post selection of contrain satisfying states. 
# so after the results, manually removing the menaingless solutions that don;t represent physical solutions, that don't respect the constraints (eg. no rotamer chosen on 1 residue, or 2 rotamers chosen)

# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian.
# Change file_path line 13
# %%
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import sys

num_rot = 3
file_path = "RESULTS/3rot_nopenalty-QAOA/11res-3rot.csv"

########################### Configure the hamiltonian from the values calculated classically with pyrosetta ############################
df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N = int(num/num_rot)
num_qubits = num

print('Qii values: \n', q)

df = pd.read_csv("energy_files/two_body_terms.csv")
value = df['E_ij'].values
Q = np.zeros((num,num))
n = 0

for j in range(0, num-3, num_rot):
    for i in range(j, j+num_rot):
        Q[i][j+3] = deepcopy(value[n])
        Q[j+3][i] = deepcopy(value[n])
        Q[i][j+4] = deepcopy(value[n+1])
        Q[j+4][i] = deepcopy(value[n+1])
        Q[i][j+5] = deepcopy(value[n+2])
        Q[j+5][i] = deepcopy(value[n+2])
        n += num_rot

print('\nQij values: \n', Q)

H = np.zeros((num,num))

for i in range(num):
    for j in range(num):
        if i != j:
            H[i][j] = np.multiply(0.25, Q[i][j])

for i in range(num):
    H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

print('\nH: \n', H)

k = 0
for i in range(num_qubits):
    k += 0.5 * q[i]

for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            k += 0.5 * 0.25 * Q[i][j]

print('\nk: \n', k)

# %% Brute force
import itertools
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

time_i = time.time()
print('starting brute force')
def generate_bitstrings(num_qubits):
    return [''.join(x) for x in itertools.product('01', repeat=num_qubits)]

def check_hamming(bitstring, substring_size):
    substrings = [bitstring[i:i+substring_size] for i in range(0, len(bitstring), substring_size)]
    return all(sub.count('1') == 1 for sub in substrings)

def create_circuit(bitstring):
    """Create a quantum circuit that prepares the quantum state for a given bitstring."""
    qc = QuantumCircuit(len(bitstring))
    for i, bit in enumerate(bitstring):
        if bit == '1':
            qc.x(i)
    return qc

def evaluate_energy(bitstring, operator):
    """Evaluate the energy of a bitstring using the specified operator."""
    circuit = create_circuit(bitstring)
    state = Statevector.from_instruction(circuit)
    if not isinstance(operator, Operator):
        operator = Operator(operator)
    
    expectation_value = state.expectation_value(operator).real
    return expectation_value

substring_size = num_rot

# chunk_size = 10000
# lowest_energy = float('inf')
# bitstring_with_lowest_energy = None

# bitstring_generator = generate_bitstrings(num_qubits)
# chunk_counter = 0
# total_valid_samples = 0

# while True:
#     valid_samples = []
#     for _ in range(chunk_size):
#         try:
#             bitstring = next(bitstring_generator)
#             if check_hamming(bitstring, num_rot):
#                 valid_samples.append(bitstring)
#         except StopIteration:
#             break
    
#     if not valid_samples:
#         break
    
#     flush_print(f"Processing chunk {chunk_counter}, {len(valid_samples)} valid samples")
#     total_valid_samples += len(valid_samples)
#     chunk_counter += 1

valid_samples = []
print('generating bitstrings...')
for bitstring in generate_bitstrings(num_qubits):
    if check_hamming(bitstring, num_rot):
        print('valid bitstring found')
        valid_samples.append(bitstring)

print("Valid samples found:", len(valid_samples))
lowest_energy = float('inf')
bitstring_with_lowest_energy = None

for bitstring in valid_samples:
    spins = [1 if bit == '0' else -1 for bit in bitstring]
    energy = 0

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                energy += 0.5 * H[i][j] * spins[i] * spins[j]

    for i in range(num_qubits):
        energy +=  H[i][i] * spins[i]

    print(f"Bitstring: {bitstring}, Value: {energy + k}")

    if energy < lowest_energy:
        lowest_energy = energy
        bitstring_with_lowest_energy = bitstring

time_f = time.time()
elapsed_time_b = time_f - time_i

print("Bitstring with lowest energy:", bitstring_with_lowest_energy)
print("Ground state energy", lowest_energy + k)

data = {
    "Experiment": ["Brute Force 3rot QAOA"],
    "Ground State Energy": [lowest_energy + k],
    "Best Measurement": [bitstring_with_lowest_energy],
    "Execution Time (seconds)": [elapsed_time_b],
    "Number of qubits": [num_qubits]
}

df = pd.DataFrame(data)
df.to_csv(file_path, index=False)


# %%
