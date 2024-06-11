# Point 1 of constraint studies for paper, Ising model with no penalties, constriants enforced by post selection of contrain satisfying states. 
# so after the results, manually removing the menaingless solutions that don;t represent physical solutions, that don't respect the constraints (eg. no rotamer chosen on 1 residue, or 2 rotamers chosen)

# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# %%
import numpy as np
import pandas as pd
import time
from copy import deepcopy

num_rot = 2
file_path = "RESULTS/nopenalty-QAOA/4res-2rot.csv"


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

for i in range(0, num-2):
    if i%2 == 0:
        Q[i][i+2] = deepcopy(value[n])
        Q[i+2][i] = deepcopy(value[n])
        Q[i][i+3] = deepcopy(value[n+1])
        Q[i+3][i] = deepcopy(value[n+1])
        n += 2
    elif i%2 != 0:
        Q[i][i+1] = deepcopy(value[n])
        Q[i+1][i] = deepcopy(value[n])
        Q[i][i+2] = deepcopy(value[n+1])
        Q[i+2][i] = deepcopy(value[n+1])
        n += 2

print('\nQij values: \n', Q)

H = np.zeros((num,num))

for i in range(num):
    for j in range(num):
        if i != j:
            H[i][j] = np.multiply(0.25, Q[i][j])

for i in range(num):
    H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

print('\nH: \n', H)

# %% Brute force
import itertools
from qiskit.quantum_info import Statevector, Operator
from qiskit import QuantumCircuit

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
possible_bitstrings = [''.join(x) for x in itertools.product('01', repeat=num_qubits)]
print("Total samples:", len(possible_bitstrings))

valid_samples = []
for bitstring in possible_bitstrings: 
    print("Checking formatted bitstring:", bitstring)
    if check_hamming(bitstring, substring_size):
        valid_samples.append((bitstring))

print("Valid samples found:", len(valid_samples))

k = 0
for i in range(num_qubits):
    k += 0.5 * q[i]

for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            k += 0.5 * 0.25 * Q[i][j]

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

print("Bitstring with lowest energy:", bitstring_with_lowest_energy)
print("Ground state energy", lowest_energy + k)
print("ground state energy without k", lowest_energy)

data = {
    "Experiment": ["Aer Simulation XY QAOA"],
    "Ground State Energy": [lowest_energy + k],
    "Best Measurement": [bitstring_with_lowest_energy],
    "Execution Time (seconds)": ["N/A"],
    "Number of qubits": [num_qubits]
}

df = pd.DataFrame(data)
df.to_csv(file_path, index=False)

# # %% ################################################ Classical optimisation ###########################################################
# from scipy.sparse.linalg import eigsh

# Z_matrix = np.array([[1, 0], [0, -1]])
# identity = np.eye(2)

# def construct_operator(qubit_indices, num_qubits):
#     operator = np.eye(1)
#     for qubit in range(num_qubits):
#         if qubit in qubit_indices:
#             operator = np.kron(operator, Z_matrix)
#         else:
#             operator = np.kron(operator, identity)
#     return operator

# C = np.zeros((2**num_qubits, 2**num_qubits))

# for i in range(num_qubits):
#     operator = construct_operator([i], num_qubits)
#     C += H[i][i] * operator

# for i in range(num_qubits):
#     for j in range(i+1, num_qubits):
#         operator = construct_operator([i, j], num_qubits)
#         C += H[i][j] * operator

# print('C :\n', C)

# # Extract the ground state energy and wavefunction
# # using sparse representation so as to be able to generalise to larger systems
# eigenvalues, eigenvectors = eigsh(C, k=num, which='SA')
# print("\n\nClassical optimisation results. \n")
# print("Ground energy eigsh: ", eigenvalues[0])
# print("ground state wavefuncion eigsh: ", eigenvectors[:,0])
# print('\n\n')

# with open(file_path, "a") as file:
#     file.write("\n\nClassical optimisation results.\n")
#     file.write(f"Ground energy eigsh: {eigenvalues[0]}\n")
#     file.write(f"Ground state wavefunction eigsh: {eigenvectors[:,0]}\n")

# # %% ############################################ Quantum optimisation ########################################################################
# from qiskit_algorithms.minimum_eigensolvers import QAOA
# from qiskit.quantum_info.operators import Pauli, SparsePauliOp
# from qiskit_algorithms.optimizers import COBYLA
# from qiskit.primitives import Sampler

# def X_op(i, num_qubits):
#     """Return an X Pauli operator on the specified qubit in a num-qubit system."""
#     op_list = ['I'] * num_qubits
#     op_list[i] = 'X'
#     return SparsePauliOp(Pauli(''.join(op_list)))

# def generate_pauli_zij(n, i, j):
#     if i<0 or i >= n or j<0 or j>=n:
#         raise ValueError(f"Indices out of bounds for n={n} qubits. ")
        
#     pauli_str = ['I']*n

#     if i == j:
#         pauli_str[i] = 'Z'
#     else:
#         pauli_str[i] = 'Z'
#         pauli_str[j] = 'Z'

#     return Pauli(''.join(pauli_str))

# q_hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

# for i in range(num_qubits):
#     for j in range(i+1, num_qubits):
#         if H[i][j] != 0:
#             pauli = generate_pauli_zij(num_qubits, i, j)
#             op = SparsePauliOp(pauli, coeffs=[H[i][j]])
#             q_hamiltonian += op

# for i in range(num_qubits):
#     pauli = generate_pauli_zij(num_qubits, i, i)
#     Z_i = SparsePauliOp(pauli, coeffs=[H[i][i]])
#     q_hamiltonian += Z_i

# def format_sparsepauliop(op):
#     terms = []
#     labels = [pauli.to_label() for pauli in op.paulis]
#     coeffs = op.coeffs
#     for label, coeff in zip(labels, coeffs):
#         terms.append(f"{coeff:.10f} * {label}")
#     return '\n'.join(terms)

# print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))

# #the mixer in QAOA should be a quantum operator representing transitions between configurations
# mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
# p = 1  # Number of QAOA layers
# initial_point = np.ones(2 * p)
# # %%
# start_time = time.time()
# qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
# result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
# end_time = time.time()

# print("\n\nThe result of the quantum optimisation using QAOA is: \n")
# print('best measurement', result.best_measurement)
# elapsed_time = end_time - start_time
# print(f"Local Simulation run time: {elapsed_time} seconds")
# print('\n\n')

# with open(file_path, "a") as file:
#     file.write("\n\nThe result of the quantum optimisation using QAOA is: \n")
#     file.write(f"'best measurement' {result.best_measurement}\n")
#     file.write(f"Local Simulation run time: {elapsed_time} seconds\n")

# # %% ############################################ Simulators ##########################################################################
# from qiskit_aer import Aer
# from qiskit_ibm_provider import IBMProvider
# from qiskit_aer.noise import NoiseModel
# from qiskit_aer.primitives import Sampler
# from qiskit.primitives import Sampler, BackendSampler
# from qiskit.transpiler import PassManager

# simulator = Aer.get_backend('qasm_simulator')
# provider = IBMProvider()
# available_backends = provider.backends()
# print("Available Backends:", available_backends)
# device_backend = provider.get_backend('ibm_torino')
# noise_model = NoiseModel.from_backend(device_backend)

# options= {
#     "noise_model": noise_model,
#     "basis_gates": simulator.configuration().basis_gates,
#     "coupling_map": simulator.configuration().coupling_map,
#     "seed_simulator": 42,
#     "shots": 1000,
#     "optimization_level": 3,
#     "resilience_level": 0
# }

# noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())

# start_time1 = time.time()
# qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
# result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
# end_time1 = time.time()

# # %%
# print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
# print('best measurement', result1.best_measurement)
# print('Optimal parameters: ', result1.optimal_parameters)
# print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
# elapsed_time1 = end_time1 - start_time1
# print(f"Aer Simulator run time: {elapsed_time1} seconds")
# print('\n\n')

# with open(file_path, "a") as file:
#     file.write("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
#     file.write(f"'best measurement' {result1.best_measurement}")
#     file.write(f"Optimal parameters: {result1.optimal_parameters}")
#     file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result1.best_measurement['value'])}")
#     file.write(f"Aer Simulator run time: {elapsed_time1} seconds")


# # %% ############################################# Hardware with QAOAAnastz ##################################################################
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
# # %%
# # real_coupling_map = backend.configuration().coupling_map
# # coupling_map = CouplingMap(couplinglist=real_coupling_map)

# def generate_linear_coupling_map(num_qubits):

#     coupling_list = [[i, i + 1] for i in range(num_qubits - 1)]
    
#     return CouplingMap(couplinglist=coupling_list)

# linear_coupling_map = generate_linear_coupling_map(num_qubits)
# # coupling_map = CouplingMap(couplinglist=[[0, 1],[0, 15], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]])
# qr = QuantumRegister(num_qubits, 'q')
# circuit = QuantumCircuit(qr)
# trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
# ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=linear_coupling_map,
#                        optimization_level=1, layout_method='trivial', routing_method='basic')
# print("\n\nAnsatz layout after explicit transpilation:", ansatz_isa._layout)

# hamiltonian_isa = q_hamiltonian.apply_layout(ansatz_isa.layout)
# print("\n\nAnsatz layout after transpilation:", hamiltonian_isa)

# # %%
# ansatz_isa.decompose().draw('mpl')

# op_counts = ansatz_isa.count_ops()
# total_gates = sum(op_counts.values())
# print("Operation counts:", op_counts)
# print("Total number of gates:", total_gates)

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
# print('The ground state energy with noisy QAOA is: ', np.real(result2.best_measurement['value']))

# # %%
# jobs = service.jobs(session_id='crk4j0r9nqa00081k6y0')

# for job in jobs:
#     if job.status().name == 'DONE':
#         results = job.result()
#         print("Job completed successfully")
# else:
#     print("Job status:", job.status())

# # %%
# from qiskit.quantum_info import Statevector, Operator

# def create_circuit(bitstring):
#     """Create a quantum circuit that prepares the quantum state for a given bitstring."""
#     qc = QuantumCircuit(len(bitstring))
#     for i, bit in enumerate(bitstring):
#         if bit == '1':
#             qc.x(i)
#     return qc

# def evaluate_energy(bitstring, operator):
#     """Evaluate the energy of a bitstring using the specified operator."""
#     circuit = create_circuit(bitstring)
#     state = Statevector.from_instruction(circuit)
#     if not isinstance(operator, Operator):
#         operator = Operator(operator)
    
#     expectation_value = state.expectation_value(operator).real
#     return expectation_value

# def get_best_measurement_from_sampler_result(sampler_result, q_hamiltonian, num_qubits):
#     if not hasattr(sampler_result, 'quasi_dists') or not isinstance(sampler_result.quasi_dists, list):
#         raise ValueError("SamplerResult does not contain 'quasi_dists' as a list")

#     best_bitstring = None
#     lowest_energy = float('inf')
#     highest_probability = -1

#     for quasi_distribution in sampler_result.quasi_dists:
#         for int_bitstring, probability in quasi_distribution.items():
#             bitstring = format(int_bitstring, '0{}b'.format(num_qubits))
#             energy = evaluate_energy(bitstring, q_hamiltonian)
#             if energy < lowest_energy:
#                 lowest_energy = energy
#                 best_bitstring = bitstring
#                 highest_probability = probability

#     return best_bitstring, highest_probability, lowest_energy


# best_bitstring, probability, value = get_best_measurement_from_sampler_result(results, q_hamiltonian, num_qubits)
# print(f"Best measurement: {best_bitstring} with ground state energy {value} and probability {probability}")

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
#     file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result2.best_measurement['value'])}")
#     file.write(f"Total Usage Time Hardware: {total_usage_time} seconds")
#     file.write(f"Total number of gates: {total_gates}\n")   

# # %%
# index = ansatz_isa.layout.final_index_layout() # Maps logical qubit index to its position in bitstring

# # measured_bitstring = result2.best_measurement['bitstring']
# measured_bitstring = best_bitstring
# original_bitstring = ['']*num_qubits

# for i, logical in enumerate(index):
#         original_bitstring[i] = measured_bitstring[logical]

# original_bitstring = ''.join(original_bitstring)
# print("Original bitstring:", original_bitstring)
