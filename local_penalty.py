# Point 2 of constraint studies for paper, Ising model with local penalties, 3 rotamers per residue

# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# Change file paths, run cells for simulations/hardware
# %%
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os
import csv
from itertools import combinations


num_res = 4
num_rot = 4
file_path = f"RESULTS/{num_rot}rot-localpenalty-QAOA/{num_res}res-{num_rot}rot.csv"
file_path_depth = f"RESULTS/Depths/{num_rot}rot-localpenalty-QAOA-noopt/{num_res}res-{num_rot}rot.csv"

########################### Configure the hamiltonian from the values calculated classically with pyrosetta ############################
df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N = int(num/num_rot)
num_qubits = num

print('Qii values: \n', q)

df2 = pd.read_csv("energy_files/two_body_terms.csv")
value = df2['E_ij'].values
Q = np.zeros((num,num))
n = 0

for j in range(0, num-num_rot, num_rot):
    for i in range(j, j+num_rot):
        for offset in range(num_rot):
            Q[i][j+num_rot+offset] = deepcopy(value[n])
            Q[j+num_rot+offset][i] = deepcopy(value[n])
            n += 1

print('\nQij values: \n', Q)

H = np.zeros((num,num))

for i in range(num):
    for j in range(num):
        if i != j:
            H[i][j] = np.multiply(0.25, Q[i][j])

for i in range(num):
    H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

print('\nH: \n', H)

# add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
def add_penalty_term(M, penalty_constant, residue_pairs):
    for i, j in residue_pairs:
        M[i][j] += penalty_constant
        M[j][i] += penalty_constant 
    return M

def generate_pairs(N, num_rot):
    pairs = []
    for i in range(0, num_rot * N, num_rot):
        # Generate all unique pairs within each num_rot-sized group
        pairs.extend((i + a, i + b) for a, b in combinations(range(num_rot), 2))
    
    return pairs

P = 6
pairs = generate_pairs(N, num_rot)
M = deepcopy(H)
M = add_penalty_term(M, P, pairs)

print("Modified Hamiltonian with Penalties:\n", M)

k = 0
for i in range(num_qubits):
    k += 0.5 * q[i]

for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            k += 0.5 * 0.25 * Q[i][j]

# %% ############################################ Quantum hamiltonian ########################################################################
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def X_op(i, num_qubits):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = ['I'] * num_qubits
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))

def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
        
    pauli_str = ['I']*n

    if i == j:
        pauli_str[i] = 'Z'
    else:
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))

q_hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

for i in range(num_qubits):
    for j in range(i+1, num_qubits):
        if M[i][j] != 0:
            pauli = generate_pauli_zij(num_qubits, i, j)
            op = SparsePauliOp(pauli, coeffs=[M[i][j]])
            q_hamiltonian += op

for i in range(num_qubits):
    pauli = generate_pauli_zij(num_qubits, i, i)
    Z_i = SparsePauliOp(pauli, coeffs=[M[i][i]])
    q_hamiltonian += Z_i

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)

print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))

# %% ############################################ q hamiltonian depth ########################################################################
import networkx as nx

file_name = f"RESULTS/qH_depth_{num_rot}rots_penalties.csv"

def pauli_shares_qubits(pauli1, pauli2):
    """
    Determines if two Pauli-Z strings share any qubits.
    If two terms share a non-identity 'Z' at the same position, they must be in different layers.
    """
    for p1, p2 in zip(pauli1, pauli2):
        if p1 == 'Z' and p2 == 'Z':
            return True  # They share a qubit
    return False  # They can be in the same layer

def compute_commuting_layers(hamiltonian):
    """
    Uses graph coloring to compute the exact number of layers for ZZ terms based on qubit overlaps.
    Separates ZZ interaction layers and single-qubit Z layers for more accurate depth estimation.

    Parameters:
    - hamiltonian: SparsePauliOp representing the Hamiltonian.

    Returns:
    - depth_HC: The total depth required for H_C.
    - num_ZZ_layers: The number of parallelizable ZZ layers.
    - num_single_Z_layers: The number of single-qubit Z layers.
    - layer_assignments: Dictionary mapping each term to its assigned layer.
    """
    pauli_labels = [pauli.to_label() for pauli in hamiltonian.paulis]

    # Separate ZZ terms and single-qubit Z terms
    zz_terms = [label for label in pauli_labels if label.count('Z') == 2]  # Two-qubit interactions
    single_z_terms = [label for label in pauli_labels if label.count('Z') == 1]  # Single-qubit rotations

    # Step 1: Create Conflict Graph (Nodes = ZZ terms, Edges = Shared Qubits)
    G = nx.Graph()
    for term in zz_terms:
        G.add_node(term)  # Each ZZ term is a node

    for i, term1 in enumerate(zz_terms):
        for j, term2 in enumerate(zz_terms):
            if i < j and pauli_shares_qubits(term1, term2):
                G.add_edge(term1, term2)  # Conflict edge (they share a qubit)

    # Step 2: Solve Graph Coloring Problem (to find minimum layers for ZZ terms)
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")

    # Step 3: Assign ZZ Layers
    num_ZZ_layers = max(coloring.values()) + 1 if coloring else 0
    layer_assignments = {term: layer for term, layer in coloring.items()}

    # Step 4: Assign Single-Qubit Z Layers (all single-qubit Z can be parallel in one layer)
    num_single_Z_layers = 1 if single_z_terms else 0
    for term in single_z_terms:
        layer_assignments[term] = num_ZZ_layers  # Put single-qubit Z terms in their own separate layer

    # Compute Corrected Depth: (ZZ layers * 3) + (Single-Z layers * 1)
    depth_HC = (num_ZZ_layers * 3) + (num_single_Z_layers * 1)

    return depth_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments

def compute_qaoa_depth(hamiltonian):
    """
    Computes the estimated depth of the QAOA circuit given a sparse Hamiltonian.
    Separates ZZ interaction layers and single-qubit Z layers.

    Parameters:
    - hamiltonian: SparsePauliOp representing H_C.

    Returns:
    - D_HC: depth of the cost Hamiltonian circuit.
    - D_QAOA_layer: depth of one full QAOA layer (H_C + H_B).
    - num_ZZ_layers: Number of ZZ layers.
    - num_single_Z_layers: Number of single-qubit Z layers.
    - layer_assignments: Layer mapping for visualization.
    """
    # Compute depth of cost Hamiltonian using Graph Coloring for ZZ layers and a separate layer for single-qubit Z terms
    D_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(hamiltonian)

    # Mixer (X-rotations) has depth 1
    D_QAOA_layer = D_HC + 1

    return D_HC, D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments

# Run the depth analysis with graph coloring
D_HC, D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_qaoa_depth(q_hamiltonian)

# Convert to DataFrame for better readability
layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
layer_df = layer_df.sort_values(by="Assigned Layer")

size = num_qubits
depth = D_QAOA_layer

file_exists = os.path.isfile(file_name)

with open(file_name, mode="a", newline="") as file:
    writer = csv.writer(file)

    # Write the header only if the file is new
    if not file_exists:
        writer.writerow(["Size", "Depth"])
    
    # Append the new result
    writer.writerow([size, depth])

# Print results
print(layer_df.to_string(index=False))  # Display the commuting layers
print(f"\n Estimated depth of H_C: {depth}")
print(f" Number of qubits: {size}")
print(f" Number of ZZ layers: {num_ZZ_layers}")
print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
print(f" Estimated depth of one QAOA layer: {D_QAOA_layer}")

 # %% ############################################ q_hamiltonian connectivity ########################################################################
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

def visualize_qubit_interactions(hamiltonian, num_qubits):
    """
    Creates a graph visualization of qubit interactions based on ZZ terms in the Hamiltonian.

    Parameters:
    - hamiltonian: SparsePauliOp representing the Hamiltonian.
    - num_qubits: Number of qubits in the system.

    Returns:
    - A plotted graph showing qubit connectivity.
    """
    pauli_labels = [pauli.to_label() for pauli in hamiltonian.paulis]

    # Initialize graph
    G = nx.Graph()

    # Add nodes for qubits
    G.add_nodes_from(range(num_qubits))

    # Identify ZZ interactions and add edges
    for label in pauli_labels:
        if label.count('Z') == 2:  # Identify ZZ terms
            qubits = [i for i, pauli in enumerate(label) if pauli == 'Z']
            if len(qubits) == 2:
                G.add_edge(qubits[0], qubits[1])

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Qubit Interaction Graph from ZZ Terms in Hamiltonian")
    plt.show()

# Run visualization on your Hamiltonian
visualize_qubit_interactions(q_hamiltonian, num_qubits)



# %% ############################################ Quantum optimisation ########################################################################

mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
p = 1  # Number of QAOA layers
initial_point = np.ones(2 * p)

# %% Local simulation, too slow when big sizes
start_time = time.time()
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
end_time = time.time()

print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result.best_measurement)
print('The ground state energy with QAOA is: ', np.real(result.best_measurement['value'] + N*P + k))
elapsed_time = end_time - start_time
print(f"Local Simulation run time: {elapsed_time} seconds")
print('\n\n')

# %% ############################################ Noisy Simulators ##########################################################################
from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit.primitives import Sampler, BackendSampler
from qiskit.transpiler import PassManager

simulator = Aer.get_backend('qasm_simulator')
provider = IBMProvider()
available_backends = provider.backends()
print("Available Backends:", available_backends)
device_backend = provider.get_backend('ibm_torino')
noise_model = NoiseModel.from_backend(device_backend)

options= {
    "noise_model": noise_model,
    "basis_gates": simulator.configuration().basis_gates,
    "coupling_map": simulator.configuration().coupling_map,
    "seed_simulator": 42,
    "shots": 5000,
    "optimization_level": 3,
    "resilience_level": 3
}

def callback(quasi_dists, parameters, energy):
    intermediate_data.append({
        'quasi_distributions': quasi_dists,
        'parameters': parameters,
        'energy': energy
    })

p = 1
noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())
intermediate_data = []

start_time1 = time.time()
qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point, callback=callback)
result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1

# %%
from qiskit_aer.primitives import Estimator
from qiskit import QuantumCircuit, transpile

def int_to_bitstring(state, total_bits):
    """Converts an integer state to a binary bitstring with padding of leading zeros."""
    return format(state, '0{}b'.format(total_bits))

def check_hamming(bitstring, substring_size):
    """Check if each substring contains exactly one '1'."""
    substrings = [bitstring[i:i+substring_size] for i in range(0, len(bitstring), substring_size)]
    return all(sub.count('1') == 1 for sub in substrings)

def calculate_bitstring_energy(bitstring, hamiltonian, backend=None):
    """
    Calculate the energy of a given bitstring for a specified Hamiltonian.

    Args:
        bitstring (str): The bitstring for which to calculate the energy.
        hamiltonian (SparsePauliOp): The Hamiltonian operator of the system, defined as a SparsePauliOp.
        backend (qiskit.providers.Backend): The quantum backend to execute circuits.

    Returns:
        float: The calculated energy of the bitstring.
    """
    # Prepare the quantum circuit for the bitstring
    num_qubits = len(bitstring)
    qc = QuantumCircuit(num_qubits)
    for i, char in enumerate(bitstring):
        if char == '1':
            qc.x(i)  # Apply X gate if the bit in the bitstring is 1
    
    # Use Aer's statevector simulator if no backend provided
    if backend is None:
        backend = Aer.get_backend('aer_simulator_statevector')

    qc = transpile(qc, backend)
    estimator = Estimator()
    resultt = estimator.run(observables=[hamiltonian], circuits=[qc], backend=backend).result()

    return resultt.values[0].real


eigenstate_distribution = result1.eigenstate
best_measurement = result1.best_measurement
final_bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

all_bitstrings = {}
for state, prob in final_bitstrings.items():
    bitstring = int_to_bitstring(state, num_qubits)
    if check_hamming(bitstring, num_rot):
        if bitstring not in all_bitstrings:
            all_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
        all_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
        energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
        all_bitstrings[bitstring]['energy'] = (all_bitstrings[bitstring]['energy'] * all_bitstrings[bitstring]['count'] + energy) / (all_bitstrings[bitstring]['count'] + 1)
        all_bitstrings[bitstring]['count'] += 1

for data in intermediate_data:
    print(f"Quasi Distribution: {data['quasi_distributions']}, Parameters: {data['parameters']}, Energy: {data['energy']}")
    for distribution in data['quasi_distributions']:
        for int_bitstring, probability in distribution.items():
            intermediate_bitstring = int_to_bitstring(int_bitstring, num_qubits)
            if check_hamming(intermediate_bitstring, num_rot):
                if intermediate_bitstring not in all_bitstrings:
                    all_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
                all_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
                energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)
                all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * all_bitstrings[intermediate_bitstring]['count'] + energy) / (all_bitstrings[intermediate_bitstring]['count'] + 1)
                all_bitstrings[intermediate_bitstring]['count'] += 1


sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])

total_bitstrings = sum(
    probability * options['shots']
    for data in intermediate_data
    for distribution in data['quasi_distributions']
    for int_bitstring, probability in distribution.items()
) + sum(
    probability * options['shots'] for state, probability in final_bitstrings.items()
)
hamming_satisfying_bitstrings = sum(bitstring_data['probability'] * options['shots'] for bitstring_data in all_bitstrings.values())
fraction_satisfying_hamming = hamming_satisfying_bitstrings / total_bitstrings
print(f"Fraction of bitstrings that satisfy the Hamming constraint: {fraction_satisfying_hamming}")

ground_state_repetition = sorted_bitstrings[0][1]['index']

print("Best Measurement:", best_measurement)
for bitstring, data in sorted_bitstrings:
    print(f"Bitstring: {bitstring}, Probability: {data['probability']}, Energy: {data['energy']}")

found = False
for bitstring, data in sorted_bitstrings:
    if bitstring == best_measurement['bitstring']:
        print('Best measurement bitstring respects Hammings conditions.\n')
        print('Ground state energy: ', data['energy']+k)
        data = {
            "Experiment": ["Aer Simulation Local Penalty QAOA"],
            "Ground State Energy": [np.real(result1.best_measurement['value'] + N*P + k)],
            "Best Measurement": [result1.best_measurement],
            "Execution Time (seconds)": [elapsed_time1],
            "Number of qubits": [num_qubits],
            "shots": [options['shots']],
            "Fraction": [fraction_satisfying_hamming],
            "Iteration Ground State": [ground_state_repetition]

        }
        found = True
        break

if not found:
    print('Best measurement bitstring does not respect Hammings conditions, take the sorted bitstring corresponding to the smallest energy.\n')
    post_selected_bitstring, post_selected_energy = sorted_bitstrings[0]
    data = {
        "Experiment": ["Aer Simulation Local Penalty QAOA, post-selected"],
        "Ground State Energy": [post_selected_energy['energy'] + N*P + k],
        "Best Measurement": [post_selected_bitstring],
        "Execution Time (seconds)": [elapsed_time1],
        "Number of qubits": [num_qubits],
        "shots": [options['shots']],
        "Fraction": [fraction_satisfying_hamming],
        "Iteration Ground State": [ground_state_repetition]

    }

df = pd.DataFrame(data)

if not os.path.isfile(file_path):
    # File does not exist, write with header
    df.to_csv(file_path, index=False)
else:
    # File exists, append without writing the header
    df.to_csv(file_path, mode='a', index=False, header=False)


# %% ############################################# Hardware with QAOAAnastz ##################################################################
# from qiskit.circuit.library import QAOAAnsatz
# from qiskit_algorithms import SamplingVQE
# from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
# from qiskit import transpile, QuantumCircuit, QuantumRegister
# from qiskit.transpiler import CouplingMap, Layout

# service = QiskitRuntimeService()
# backend = service.backend("ibm_torino")
# print('Coupling Map of hardware: ', backend.configuration().coupling_map)

# ansatz = QAOAAnsatz(q_hamiltonian, mixer_operator=mixer_op, reps=p)
# print('\n\nQAOAAnsatz: ', ansatz)

# target = backend.target
# ansatz.count_ops
# # %%
# # real_coupling_map = backend.configuration().coupling_map
# # coupling_map = CouplingMap(couplinglist=real_coupling_map)

# def generate_linear_coupling_map(num_qubits):

#     coupling_list = [[i, i + 1] for i in range(num_qubits - 1)]
    
#     return CouplingMap(couplinglist=coupling_list)

# linear_coupling_map = generate_linear_coupling_map(num_qubits)
# # coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14], [14, 13], [15, 0], [16, 4], [17, 8]])
# # coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [17, 8], [18, 12], [19, 15], [19, 20]])
# # coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22]])
# # coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [26, 25]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [26, 25], [26, 27], [27, 17], [27, 26], [27, 28], [28, 27], [28, 29], [29, 28]])
# qr = QuantumRegister(num_qubits, 'q')
# circuit = QuantumCircuit(qr)
# trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
# ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=coupling_map,
#                        optimization_level=0, layout_method='dense', routing_method='basic')
# print("\n\nAnsatz layout after explicit transpilation:", ansatz_isa._layout)

# hamiltonian_isa = q_hamiltonian.apply_layout(ansatz_isa.layout)
# print("\n\nAnsatz layout after transpilation:", hamiltonian_isa)

# # %%
# ansatz_isa.decompose().draw('mpl')

# op_counts = ansatz_isa.count_ops()
# total_gates = sum(op_counts.values())
# CNOTs = op_counts['cz']
# depth = ansatz_isa.depth()
# print("Operation counts:", op_counts)
# print("Total number of gates:", total_gates)
# print("Depth of the circuit: ", depth)

# data_depth = {
#     "Experiment": ["Hardware XY-QAOA"],
#     "Total number of gates": [total_gates],
#     "Depth of the circuit": [depth],
#     "CNOTs": [CNOTs]
# }

# df_depth = pd.DataFrame(data_depth)
# df_depth.to_csv(file_path_depth, index=False)


# # %%
# session = Session(backend=backend)
# print('\nhere 1')
# sampler = Sampler(backend=backend, session=session)
# print('here 2')
# qaoa2 = SamplingVQE(sampler=sampler, ansatz=ansatz_isa, optimizer=COBYLA(), initial_point=initial_point)
# print('here 3')
# result2 = qaoa2.compute_minimum_eigenvalue(hamiltonian_isa)

# print("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
# print('best measurement', result2.best_measurement)
# print('Optimal parameters: ', result2.optimal_parameters)
# print('The ground state energy with noisy QAOA is: ', np.real(result2.best_measurement['value']) + N*P + k)

# # %%
# jobs = service.jobs(session_id='crsn8xvx484g008f4200')

# for job in jobs:
#     if job.status().name == 'DONE':
#         results = job.result()
#         print("Job completed successfully")
# else:
#     print("Job status:", job.status())

# # %%
# total_usage_time = 0
# for job in jobs:
#     job_result = job.usage_estimation['quantum_seconds']
#     total_usage_time += job_result

# print(f"Total Usage Time Hardware: {total_usage_time} seconds")
# print('\n\n')

# with open(file_path, "a") as file:
#     file.write("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
#     file.write(f"'best measurement' {result2.best_measurement}")
#     file.write(f"Optimal parameters: {result2.optimal_parameters}")
#     file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result2.best_measurement['value']) + N*P + k}")
#     file.write(f"Total Usage Time Hardware: {total_usage_time} seconds")
#     file.write(f"Total number of gates: {total_gates}\n")   
#     file.write(f"Depth of circuit: {depth}\n")

# # %%
# index = ansatz_isa.layout.final_index_layout() # Maps logical qubit index to its position in bitstring

# measured_bitstring = result2.best_measurement['bitstring']
# original_bitstring = ['']*num_qubits

# for i, logical in enumerate(index):
#         original_bitstring[i] = measured_bitstring[logical]

# original_bitstring = ''.join(original_bitstring)
# print("Original bitstring:", original_bitstring)

# data = {
#     "Experiment": ["Quantum Optimisation (QAOA)", "Noisy Quantum Optimisation (Aer Simulator)", "Quantum Optimisation (QAOAAnsatz)"],
#     "Ground State Energy": [result.optimal_value + k, np.real(result1.best_measurement['value'] + k), np.real(result2.best_measurement['value'])],
#     "Best Measurement": [result.optimal_parameters, result1.best_measurement, result2.best_measurement],
#     "Optimal Parameters": ["N/A", "N/A", result2.optimal_parameters],
#     "Execution Time (seconds)": [elapsed_time, elapsed_time1, total_usage_time],
#     "Total Gates": ["N/A", total_gates, total_gates],
#     "Circuit Depth": ["N/A", depth, depth]
# }

# df.to_csv(file_path, index=False)