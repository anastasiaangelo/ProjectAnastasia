# Point 2 of constraint studies for paper, Ising model with local penalties

# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# %%
import numpy as np
import pandas as pd
from copy import deepcopy

num_rot = 3
file_path = "RESULTS/hardware/12res-3rot.csv"

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

# 2 rot per residue
# for i in range(0, num-2):
#     if i%2 == 0:
#         Q[i][i+2] = deepcopy(value[n])
#         Q[i+2][i] = deepcopy(value[n])
#         Q[i][i+3] = deepcopy(value[n+1])
#         Q[i+3][i] = deepcopy(value[n+1])
#         n += 2
#     elif i%2 != 0:
#         Q[i][i+1] = deepcopy(value[n])
#         Q[i+1][i] = deepcopy(value[n])
#         Q[i][i+2] = deepcopy(value[n+1])
#         Q[i+2][i] = deepcopy(value[n+1])
#         n += 2

# 3 rot per residue
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

# # add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
# def add_penalty_term(M, penalty_constant, residue_pairs):
#     for i, j in residue_pairs:
#         M[i][j] += penalty_constant
        
#     return M

# P = 6

# def generate_pairs(N):
#     pairs = [(i, i+1) for i in range(0, 2*N, 2)]
#     return pairs

# pairs = generate_pairs(N)

# M = deepcopy(H)
# M = add_penalty_term(M, P, pairs)

# %% ############################################ Quantum optimisation ########################################################################
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit import QuantumCircuit

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
        if H[i][j] != 0:
            pauli = generate_pauli_zij(num_qubits, i, j)
            op = SparsePauliOp(pauli, coeffs=[H[i][j]])
            q_hamiltonian += op

for i in range(num_qubits):
    pauli = generate_pauli_zij(num_qubits, i, i)
    Z_i = SparsePauliOp(pauli, coeffs=[H[i][i]])
    q_hamiltonian += Z_i

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)

print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))

#the mixer in QAOA should be a quantum operator representing transitions between configurations
p = 1  # Number of QAOA layers
initial_point = np.ones(2 * p)

def create_custom_xy_mixer(num_qubits):
    hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
    for i in range(0, num_qubits - 2, 3): 
        x1x2 = ['I'] * num_qubits
        y1y2 = ['I'] * num_qubits
        x2x3 = ['I'] * num_qubits
        y2y3 = ['I'] * num_qubits
        x1x3 = ['I'] * num_qubits
        y1y3 = ['I'] * num_qubits

        x1x2[i] = 'X'
        x1x2[i+1] = 'X'
        y1y2[i] = 'Y'
        y1y2[i+1] = 'Y'
        x2x3[i+1] = 'X'
        x2x3[i+2] = 'X'
        y2y3[i+1] = 'Y'
        y2y3[i+2] = 'Y'
        x1x3[i] = 'X'
        x1x3[i+2] = 'X'
        y1y3[i] = 'Y'
        y1y3[i+2] = 'Y' 

        x1x2 = SparsePauliOp(Pauli(''.join(x1x2)), coeffs=[1/2])
        y1y2 = SparsePauliOp(Pauli(''.join(y1y2)), coeffs=[1/2])
        x2x3 = SparsePauliOp(Pauli(''.join(x2x3)), coeffs=[1/2])
        y2y3 = SparsePauliOp(Pauli(''.join(y2y3)), coeffs=[1/2])
        x1x3 = SparsePauliOp(Pauli(''.join(x1x3)), coeffs=[1/2])
        y1y3 = SparsePauliOp(Pauli(''.join(y1y3)), coeffs=[1/2])

        hamiltonian += x1x2 + y1y2 + x2x3 + y2y3 + x1x3 + y1y3
    return hamiltonian

XY_mixer = create_custom_xy_mixer(num_qubits)

def generate_initial_bitstring(num_qubits):
    pattern = '100'
    bitstring = (pattern * (num_qubits // 3 + 1))[:num_qubits]
    return bitstring
# %%
initial_bitstring = generate_initial_bitstring(num_qubits)
state_vector = np.zeros(2**num_qubits)
indexx = int(initial_bitstring, 2)
state_vector[indexx] = 1
qc = QuantumCircuit(num_qubits)
qc.initialize(state_vector, range(num_qubits))

# %% ############################################# Hardware with QAOAAnastz ##################################################################
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.transpiler import Layout

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
print('Coupling Map of hardware: ', backend.configuration().coupling_map)

ansatz = QAOAAnsatz(q_hamiltonian, mixer_operator=XY_mixer, reps=p)
print('\n\nQAOAAnsatz: ', ansatz)

# opt_parameters = [3.15625, 1.0]
# ansatz_one_rep = QAOAAnsatz(q_hamiltonian, mixer_operator=XY_mixer, reps=1)
# params_dict = {param: value for param, value in zip(ansatz_one_rep.parameters, opt_parameters)}
# bound_circuit = ansatz_one_rep.assign_parameters(params_dict)

target = backend.target

# %%
filtered_coupling_map = [coupling for coupling in backend.configuration().coupling_map if coupling[0] < num_qubits and coupling[1] < num_qubits]

qr = QuantumRegister(num_qubits, 'q')
circuit = QuantumCircuit(qr)
trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})

ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=filtered_coupling_map,
                       optimization_level= 3, layout_method='dense', routing_method='stochastic')
print("\n\nAnsatz layout after explicit transpilation:", ansatz_isa._layout)

hamiltonian_isa = q_hamiltonian.apply_layout(ansatz_isa.layout)
print("\n\nAnsatz layout after transpilation:", hamiltonian_isa)

# ansatz_one_rep_isa = transpile(ansatz_one_rep, backend=backend, initial_layout=trivial_layout, coupling_map=filtered_coupling_map,
#                                optimization_level=3, layout_method='trivial', routing_method='stochastic')
# hamiltonian_isa_one_rep = q_hamiltonian.apply_layout(ansatz_one_rep_isa.layout)
# print("\n\nAnsatz layout after transpilation:", hamiltonian_isa_one_rep)

# %%
jobs = service.jobs(session_id='csawz5m7yykg0082qj6g')

for job in jobs:
    if job.status().name == 'DONE':
        results = job.result()
        print("Job completed successfully")
else:
    print("Job status:", job.status())

# %%
from qiskit_aer.primitives import Estimator
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

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
    num_qubits_qc = len(bitstring)
    qc = QuantumCircuit(num_qubits_qc)
    for i, char in enumerate(bitstring):
        if char == '1':
            qc.x(i)  # Apply X gate if the bit in the bitstring is 1
    
    # Use Aer's statevector simulator if no backend provided
    if backend is None:
        backend = Aer.get_backend('aer_simulator_statevector')

    qc = transpile(qc, backend=backend, coupling_map=filtered_coupling_map)
    estimator = Estimator()
    resultt = estimator.run(observables=[hamiltonian], circuits=[qc], backend=backend).result()

    return resultt.values[0].real

def get_best_measurement_from_sampler_result(sampler_result, num_qubits, num_ancillas):
    logical_qubits = num_qubits - num_ancillas
    print('logical qubits', logical_qubits)
    if not hasattr(sampler_result, 'quasi_dists') or not isinstance(sampler_result.quasi_dists, list):
        raise ValueError("SamplerResult does not contain 'quasi_dists' as a list")

    best_bitstring = None
    lowest_energy = float('inf')
    highest_probability = -1
    total_bitstrings = 0
    valid_bitstrings = 0

    for quasi_distribution in sampler_result.quasi_dists:
        for state, probability in quasi_distribution.items():
            bitstring = int_to_bitstring(state, num_qubits)
            # logical_bitstring = bitstring[:logical_qubits]
            total_bitstrings += 1
            if check_hamming(bitstring, num_rot):
                energy = calculate_bitstring_energy(bitstring, q_hamiltonian, backend)
                print(f"Bitstring: {bitstring}, Energy: {energy}, Probability: {probability}")
                valid_bitstrings += 1
                if energy < lowest_energy:
                    lowest_energy = energy
                    best_bitstring = bitstring
                    highest_probability = probability

    return best_bitstring, highest_probability, lowest_energy, total_bitstrings, valid_bitstrings

num_ancillas = ansatz_isa.num_qubits - num_qubits
best_bitstring, probability, value, total_bitstrings, valid_bitstrings = get_best_measurement_from_sampler_result(results, ansatz_isa.num_qubits, num_ancillas)
fraction_satisfying_hamming = valid_bitstrings / total_bitstrings
print(f"Best measurement: {best_bitstring} with ground state energy {value+k} and probability {probability}")
print(f"Fraction of bitstrings that satisfy the Hamming constraint: {fraction_satisfying_hamming}")
# %%
total_usage_time = 0
for job in jobs:
    job_result = job.usage_estimation['quantum_seconds']
    total_usage_time += job_result

print(f"Total Usage Time Hardware: {total_usage_time} seconds")
print('\n\n')

data = {
        "Experiment": ["Hardware simulation QAOAAnsatz"],
        "Ground State Energy": [value+k],
        "Best Measurement": [best_bitstring],
        "Execution Time (seconds)": [total_usage_time],
        "Number of qubits": [num_qubits], 
        "Fraction of bitstrings that satisfy the Hamming constraint": [fraction_satisfying_hamming]
}
df = pd.DataFrame(data)
df.to_csv(file_path, index=False)

# %%
# index = ansatz_isa.layout.final_index_layout() # Maps logical qubit index to its position in bitstring

# measured_bitstring = best_bitstring
# original_bitstring = ['']*num_qubits

# for i, logical in enumerate(index):
#         original_bitstring[i] = measured_bitstring[logical]

# original_bitstring = ''.join(original_bitstring)
# print("Original bitstring:", original_bitstring)
