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
import itertools
from itertools import combinations, product
import matplotlib.pyplot as plt


num_res = 4
num_rot = 2
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

P = 1.5
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

# %% ############################################ classical energy landscape ########################################################################
import itertools

J = np.zeros((num,num))
h = np.zeros(num)

# Interaction matrix
for i in range(num):
    for j in range(num):
        if i != j:
            J[i][j] = np.multiply(0.25, Q[i][j])

# External field
for i in range(num):
    h[i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

# Generate all possible spin states (-1 or +1)
spin_states = list(itertools.product([-1, 1], repeat=len(h)))

# Compute energy for each configuration
def hamiltonian(spins):
    spins = np.array(spins)
    return -0.5 * np.dot(spins, np.dot(J, spins)) - np.dot(h, spins)

energies = [hamiltonian(state) for state in spin_states]

# Plot energy levels
plt.figure(figsize=(8,6))
plt.bar(range(len(spin_states)), energies, tick_label=[str(s) for s in spin_states])
plt.xlabel("Spin Configurations")
plt.ylabel("Energy")
plt.title("Energy Landscape of the Hamiltonian")
plt.xticks(rotation=45)
plt.show()



#  %% ################################################ Classical optimisation ###########################################################
from scipy.sparse.linalg import eigsh

Z_matrix = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

def construct_operator(qubit_indices, num_qubits):
    operator = np.eye(1)
    for qubit in range(num_qubits):
        if qubit in qubit_indices:
            operator = np.kron(operator, Z_matrix)
        else:
            operator = np.kron(operator, identity)
    return operator

C = np.zeros((2**num_qubits, 2**num_qubits))

for i in range(num_qubits):
    operator = construct_operator([i], num_qubits)
    C += H[i][i] * operator

for i in range(num_qubits):
    for j in range(i+1, num_qubits):
        operator = construct_operator([i, j], num_qubits)
        C += H[i][j] * operator

print('C :\n', C)

def create_hamiltonian(pairs, P, num_qubits):
    H_pen = np.zeros((2**num_qubits, 2**num_qubits))
    def tensor_term(term_indices):
        term = [Z_matrix if i in term_indices else identity for i in range(num_qubits)]
        result = term[0]
        for t in term[1:]:
            result = np.kron(result, t)
        return result
    
    for pair in pairs:
        term = tensor_term(pair)
        H_pen += P * term

    return H_pen

H_penalty = create_hamiltonian(pairs, P, num_qubits)
H_tot = C + H_penalty

# Extract the ground state energy and wavefunction
eigenvalues, eigenvectors = eigsh(H_tot, k=num, which='SA')
print("\n\nClassical optimisation results. \n")
print("Ground energy eigsh: ", eigenvalues[0] + N*P + k)
print("ground state wavefuncion eigsh: ", eigenvectors[:,0])
print('\n\n')


# %% Plot energy spectrum
plt.figure(figsize=(8,6))
plt.plot(range(len(eigenvalues)), sorted(eigenvalues), 'bo-', label="Eigenvalues")
plt.xlabel("Index")
plt.ylabel("Energy")
plt.title("Eigenvalue Spectrum of the Hamiltonian")
plt.legend()
plt.show()



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

# %% ############################################ energy landscape ########################################################################
from qiskit.quantum_info import Statevector

# Define a quantum state with two parameters
def quantum_state(theta, phi):
    return Statevector.from_label('000000') * np.cos(theta) + np.exp(1j * phi) * Statevector.from_label('111111') * np.sin(theta)

# Generate grid points
theta_vals = np.linspace(0, np.pi, 50)
phi_vals = np.linspace(0, 2*np.pi, 50)
Theta, Phi = np.meshgrid(theta_vals, phi_vals)

# Compute expectation values
Energies = np.array([
    np.real(quantum_state(t, p).expectation_value(q_hamiltonian))
    for t, p in zip(np.ravel(Theta), np.ravel(Phi))
]).reshape(Theta.shape)

# Plot as a contour map
plt.figure(figsize=(8,6))
plt.contourf(Theta, Phi, Energies, levels=30, cmap='viridis')
plt.colorbar(label="Energy Expectation")
plt.xlabel("Theta")
plt.ylabel("Phi")
plt.title("Energy Landscape (Quantum Hamiltonian)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from qiskit.quantum_info import SparsePauliOp


# %%
from qiskit.quantum_info import Statevector

# Define a parameterized quantum state
def quantum_state(theta, N):
    zero_state = "0" * N  # |000...0⟩
    one_state = "1" * N   # |111...1⟩
    return Statevector.from_label(zero_state) * np.cos(theta) + Statevector.from_label(one_state) * np.sin(theta)

# Compute expectation values
theta_vals = np.linspace(0, np.pi, 100)
energies = [np.real(quantum_state(theta, num_qubits).expectation_value(q_hamiltonian)) for theta in theta_vals]

# Fit to a double-well shape
y_vals = (theta_vals**4 - 3*theta_vals**2) * max(energies) / max(theta_vals**4 - 3*theta_vals**2)

# Plot
plt.figure(figsize=(8,6))
plt.plot(theta_vals, energies, 'k', linewidth=2, label="Energy Landscape")
plt.xlabel("Variational Parameter (θ)")
plt.ylabel("Energy Expectation")
plt.title("Quantum Energy Landscape")
plt.legend()
plt.show()

# %% ############################################ energy landscape ########################################################################
from qiskit.primitives import Estimator
from qiskit.circuit import Parameter, QuantumCircuit
from scipy.signal import argrelextrema  
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch
from scipy.interpolate import make_interp_spline

theta = Parameter("θ")

def variational_circuit():
    """Creates a more expressive variational ansatz with entanglement."""
    qc = QuantumCircuit(num_qubits)
    
    for qubit in range(num_qubits):
        qc.ry(theta, qubit)

    for qubit in range(num_qubits - 1):
        qc.cx(qubit, qubit + 1) 

    return qc

estimator = Estimator()

theta_vals = np.linspace(0, np.pi, 50)  # Adjust granularity as needed
energy_vals = []

for val in theta_vals:
    qc = variational_circuit() 
    bound_qc = qc.assign_parameters({theta: val})  
    
    job = estimator.run([bound_qc], [q_hamiltonian])
    energy_vals.append(job.result().values[0])  

energy_vals = np.array(energy_vals) 
theta_vals = np.array(theta_vals)   

local_minima_indices = argrelextrema(energy_vals, np.less)[0]  
theta_minima = theta_vals[local_minima_indices]
energy_minima = energy_vals[local_minima_indices]

theta_minima = np.insert(theta_minima, 0, 0)
energy_minima = np.insert(energy_minima, 0, energy_vals[0])

local_maxima_indices = argrelextrema(energy_vals, np.greater)[0]
theta_transition = None
energy_transition = None

if len(local_maxima_indices) > 0: 
    theta_transition = theta_vals[local_maxima_indices[0]]  
    energy_transition = energy_vals[local_maxima_indices[0]]

global_min_index = np.argmin(energy_vals)
global_min_theta = theta_vals[global_min_index]
global_min_energy = energy_vals[global_min_index]

plt.figure(figsize=(10, 5))
plt.plot(theta_vals, energy_vals, 'k', linewidth=1, label="Energy Landscape")

if len(theta_minima) > 0:
    plt.scatter(theta_minima, energy_minima, color='lightskyblue', edgecolor='steelblue', s=80, label="Stable States")

if theta_transition is not None:
    plt.scatter(theta_transition, energy_transition, color='royalblue', edgecolor='blue', s=80, label="Transition State")

plt.scatter(global_min_theta, global_min_energy, color='lime', edgecolor='seagreen', 
            s=100, label="Global Minimum", zorder=3)


# Add transition path arrow (from lowest local minimum to transition state)
# verts = []  # Start at first local minimum

# if len(theta_minima) > 0:
#     verts.append((theta_minima[0], energy_minima[0]))  # Start at local minimum
#     if theta_transition is not None:
#         verts.append((theta_transition, energy_transition + 5))  # Pass through local max
# verts.append((global_min_theta, global_min_energy))  # End at global minimum

# if len(verts) > 1:  # Ensure there are enough points for a curve
#     codes = [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1)
#     path = Path(verts, codes)
#     patch = PathPatch(path, linestyle="dashed", edgecolor="limegreen", linewidth=1.5, fill=False)
#     plt.gca().add_patch(patch)


# theta_start = theta_vals[0]
# energy_start = energy_vals[0]

# if theta_transition is not None:
#     transition_indices = np.logical_and(theta_vals >= theta_minima[0], theta_vals <= global_min_theta)
#     theta_curve = theta_vals[transition_indices]
#     energy_curve = energy_vals[transition_indices]

#     if len(theta_curve) > 2:  # Ensure enough points for a spline
#         spline = make_interp_spline(theta_curve, energy_curve, k=min(2, len(theta_curve)-1))  # Reduce k if needed
#         theta_smooth = np.linspace(theta_curve.min(), theta_curve.max(), 100)
#         energy_smooth = spline(theta_smooth)

#         # Plot the smooth transition path
#         plt.plot(theta_smooth, energy_smooth, linestyle="dotted", color="limegreen", linewidth=1.5)

#         # Add arrow at the end
#         arrow = FancyArrowPatch(
#             (theta_smooth[-2], energy_smooth[-2]), (theta_smooth[-1], energy_smooth[-1]), 
#             arrowstyle="->", color="limegreen", mutation_scale=12, linewidth=1.5
#         )
#         plt.gca().add_patch(arrow)

plt.xlabel("Variational Parameter (θ)")
plt.ylabel("Energy Expectation Value")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.savefig(f"energy_landscape_{num_qubits}q_P={P}_2.pdf")
plt.show()

# %% ############################################ q hamiltonian depth ########################################################################
import networkx as nx
mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))

file_name = f"RESULTS/qH_depths/total_depth_{num_rot}rots_penalties.csv"

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


def compute_qaoa_depth(cost_hamiltonian, mixer_hamiltonian=None, mixer_type="X"):
    """
    Computes the estimated depth of a full QAOA layer.

    Parameters:
    - cost_hamiltonian: SparsePauliOp for H_C.
    - mixer_hamiltonian: SparsePauliOp for H_B (if XY).
    - mixer_type: "X" or "XY".

    Returns:
    - D_HC: depth of the cost Hamiltonian circuit.
    - D_QAOA_layer: total depth of one QAOA layer.
    - details: dictionary with breakdown of depths and layers.
    """

    # Cost Hamiltonian depth
    D_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(cost_hamiltonian)

    if mixer_type == "X":
        D_HB = 1
        num_mixer_layers = 1
    elif mixer_type == "XY":
        if mixer_hamiltonian is None:
            raise ValueError("XY mixer selected but no mixer Hamiltonian provided.")
        D_HB, num_XY_layers, _, _ = compute_commuting_layers(mixer_hamiltonian)
        num_mixer_layers = num_XY_layers
    else:
        raise ValueError("Unknown mixer type.")

    D_QAOA_layer = D_HC + D_HB

    details = {
        "D_HC": D_HC,
        "D_HB": D_HB,
        "D_QAOA_layer": D_QAOA_layer,
        "num_ZZ_layers": num_ZZ_layers,
        "num_single_Z_layers": num_single_Z_layers,
        "num_mixer_layers": num_mixer_layers,
        "mixer_type": mixer_type,
    }

    return D_HC, D_QAOA_layer, layer_assignments, details

# Run the depth analysis with graph coloring
D_HC, D_QAOA_layer, layer_assignments, details = compute_qaoa_depth(q_hamiltonian, mixer_op, "X")

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
# print(f" Number of ZZ layers: {num_ZZ_layers}")
# print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
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
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms import SamplingVQE
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.transpiler import CouplingMap, Layout

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
print('Coupling Map of hardware: ', backend.configuration().coupling_map)

ansatz = QAOAAnsatz(q_hamiltonian, mixer_operator=mixer_op, reps=p)
print('\n\nQAOAAnsatz: ', ansatz)

target = backend.target
ansatz.count_ops
# %%
# real_coupling_map = backend.configuration().coupling_map
# coupling_map = CouplingMap(couplinglist=real_coupling_map)
filtered_coupling_map = [coupling for coupling in backend.configuration().coupling_map if coupling[0] < num_qubits and coupling[1] < num_qubits]

def generate_linear_coupling_map(num_qubits):

    coupling_list = [[i, i + 1] for i in range(num_qubits - 1)]
    
    return CouplingMap(couplinglist=coupling_list)

linear_coupling_map = generate_linear_coupling_map(num_qubits)
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14], [14, 13], [15, 0], [16, 4], [17, 8]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [17, 8], [18, 12], [19, 15], [19, 20]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [26, 25]])
coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [26, 25], [26, 27], [27, 17], [27, 26], [27, 28], [28, 27], [28, 29], [29, 28]])
qr = QuantumRegister(num_qubits, 'q')
circuit = QuantumCircuit(qr)
trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=coupling_map,
                       optimization_level=0, layout_method='dense', routing_method='basic')
print("\n\nAnsatz layout after explicit transpilation:", ansatz_isa._layout)

hamiltonian_isa = q_hamiltonian.apply_layout(ansatz_isa.layout)
print("\n\nAnsatz layout after transpilation:", hamiltonian_isa)

# %%
ansatz_isa.decompose().draw('mpl')

op_counts = ansatz_isa.count_ops()
total_gates = sum(op_counts.values())
CNOTs = op_counts['cz']
depth = ansatz_isa.depth()
print("Operation counts:", op_counts)
print("Total number of gates:", total_gates)
print("Depth of the circuit: ", depth)

data_depth = {
    "Experiment": ["Hardware XY-QAOA"],
    "Total number of gates": [total_gates],
    "Depth of the circuit": [depth],
    "CNOTs": [CNOTs]
}

df_depth = pd.DataFrame(data_depth)
df_depth.to_csv(file_path_depth, index=False)


# %%
session = Session(backend=backend)
print('\nhere 1')
sampler = Sampler(backend=backend, session=session)
print('here 2')
qaoa2 = SamplingVQE(sampler=sampler, ansatz=ansatz_isa, optimizer=COBYLA(), initial_point=initial_point)
print('here 3')
result2 = qaoa2.compute_minimum_eigenvalue(hamiltonian_isa)

print("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
print('best measurement', result2.best_measurement)
print('Optimal parameters: ', result2.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result2.best_measurement['value']) + N*P + k)

# %%
jobs = service.jobs(session_id='crsn8xvx484g008f4200')

for job in jobs:
    if job.status().name == 'DONE':
        results = job.result()
        print("Job completed successfully")
else:
    print("Job status:", job.status())

# %%
total_usage_time = 0
for job in jobs:
    job_result = job.usage_estimation['quantum_seconds']
    total_usage_time += job_result

print(f"Total Usage Time Hardware: {total_usage_time} seconds")
print('\n\n')

with open(file_path, "a") as file:
    file.write("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
    file.write(f"'best measurement' {result2.best_measurement}")
    file.write(f"Optimal parameters: {result2.optimal_parameters}")
    file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result2.best_measurement['value']) + N*P + k}")
    file.write(f"Total Usage Time Hardware: {total_usage_time} seconds")
    file.write(f"Total number of gates: {total_gates}\n")   
    file.write(f"Depth of circuit: {depth}\n")

# %%
index = ansatz_isa.layout.final_index_layout() # Maps logical qubit index to its position in bitstring

measured_bitstring = result2.best_measurement['bitstring']
original_bitstring = ['']*num_qubits

for i, logical in enumerate(index):
        original_bitstring[i] = measured_bitstring[logical]

original_bitstring = ''.join(original_bitstring)
print("Original bitstring:", original_bitstring)

data = {
    "Experiment": ["Quantum Optimisation (QAOA)", "Noisy Quantum Optimisation (Aer Simulator)", "Quantum Optimisation (QAOAAnsatz)"],
    "Ground State Energy": [result.optimal_value + k, np.real(result1.best_measurement['value'] + k), np.real(result2.best_measurement['value'])],
    "Best Measurement": [result.optimal_parameters, result1.best_measurement, result2.best_measurement],
    "Optimal Parameters": ["N/A", "N/A", result2.optimal_parameters],
    "Execution Time (seconds)": [elapsed_time, elapsed_time1, total_usage_time],
    "Total Gates": ["N/A", total_gates, total_gates],
    "Circuit Depth": ["N/A", depth, depth]
}

df.to_csv(file_path, index=False)