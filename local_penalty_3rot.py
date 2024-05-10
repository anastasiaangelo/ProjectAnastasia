# Point 2 of constraint studies for paper, Ising model with local penalties

# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# %%
import numpy as np
import pandas as pd
import time
from copy import deepcopy

num_rot = 3
file_path = "RESULTS/3rot-localpenalty-QAOA/7res-3rot.csv"
file_path_depth = "RESULTS/Depths/3rot-localpenalty-QAOA/7res-3rot.csv"

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

for i in range(0, num):
    if n + 1 < len(value):
        if i%2 == 0 and i + 3 < num:
            Q[i][i+2] = deepcopy(value[n])
            Q[i+2][i] = deepcopy(value[n])
            Q[i][i+3] = deepcopy(value[n+1])
            Q[i+3][i] = deepcopy(value[n+1])
            n += 2
        elif i%2 != 0 and i + 2 < num:
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

# add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
def add_penalty_term(M, penalty_constant, residue_pairs):
    for i, j in residue_pairs:
        M[i][j] += penalty_constant
        M[j][i] += penalty_constant  # Symmetrically update the matrix
    return M

def generate_pairs(N):
    pairs = []
    for i in range(0, 3*N, 3):
        pairs.append((i, i+1))   # Pair (x1, x2)
        pairs.append((i, i+2))   # Pair (x1, x3)
        pairs.append((i+1, i+2)) # Pair (x2, x3)
    return pairs


P = 6
pairs = generate_pairs(N)
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


# %% ################################################ Classical optimisation ###########################################################
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

# %% ############################################ Quantum optimisation ########################################################################
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
    "shots": 1000,
    "optimization_level": 3,
    "resilience_level": 0
}

noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())

start_time1 = time.time()
qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
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

bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

bitstring_data = {}
for state, prob in bitstrings.items():
    bitstring = int_to_bitstring(state, num_qubits)
    if check_hamming(bitstring, num_rot):
        energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
        bitstring_data[bitstring] = {'probability': prob, 'energy': energy}

sorted_bitstrings = sorted(bitstring_data.items(), key=lambda x: x[1]['energy'])

print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
for bitstring, data in sorted_bitstrings:
    print(f"Bitstring: {bitstring}, Probability: {data['probability']}, Energy: {data['energy']}")

found = False
for bitstring, data in sorted_bitstrings:
    if bitstring == best_measurement['bitstring']:
        print('Best measurement bitstring respects Hammings conditions.\n')
        print('Ground state energy: ', data['energy']+N*P+k)
        data = {
            "Experiment": ["Aer Simulation local penalty QAOA"],
            "Ground State Energy": [np.real(result1.best_measurement['value'] + N*P + k)],
            "Best Measurement": [result1.best_measurement],
            "Execution Time (seconds)": [elapsed_time1],
            "Number of qubits": [num_qubits]
}
        found = True
        break

if not found:
    print('Best measurement bitstring does not respect Hammings conditions, take the sorted bitstring corresponding to the smallest energy.\n')
    post_selected_bitstring, post_selected_energy = sorted_bitstrings[0]
    data = {
        "Experiment": ["Aer Simulation local penalty QAOA, post-selected"],
        "Ground State Energy": [post_selected_energy['energy'] + N*P + k],
        "Best Measurement": [post_selected_bitstring],
        "Execution Time (seconds)": [elapsed_time1],
        "Number of qubits": [num_qubits]
    }

df = pd.DataFrame(data)
df.to_csv(file_path, index=False)

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
# %%
# real_coupling_map = backend.configuration().coupling_map
# coupling_map = CouplingMap(couplinglist=real_coupling_map)

def generate_linear_coupling_map(num_qubits):

    coupling_list = [[i, i + 1] for i in range(num_qubits - 1)]
    
    return CouplingMap(couplinglist=coupling_list)

# linear_coupling_map = generate_linear_coupling_map(num_qubits)
# coupling_map = CouplingMap(couplinglist=[[0, 1],[0, 15], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]])
coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14], [14, 13], [15, 0], [16, 4], [17, 8]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [17, 8], [18, 12], [19, 15]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [25, 35], [26, 25], [26, 27], [27, 17], [27, 26]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [26, 25], [26, 27], [27, 17], [27, 26], [27, 28], [28, 27], [28, 29], [29, 28]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 2], [2, 3], [3, 4], [4, 5], [4, 16], [5, 6], [6, 7], [7, 8], [8, 9], [8, 17], [9, 10], [10, 11], [11, 12], [12, 13], [12, 18], [13, 14], [15, 19], [16, 23], [17, 27], [18, 31], [19, 20], [20, 21], [21, 22], [21, 34], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27]])
qr = QuantumRegister(num_qubits, 'q')
circuit = QuantumCircuit(qr)
trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=coupling_map,
                       optimization_level= 3, layout_method='dense', routing_method='stochastic')
print("\n\nAnsatz layout after explicit transpilation:", ansatz_isa._layout)

hamiltonian_isa = q_hamiltonian.apply_layout(ansatz_isa.layout)
print("\n\nAnsatz layout after transpilation:", hamiltonian_isa)

# %%
ansatz_isa.decompose().draw('mpl')

op_counts = ansatz_isa.count_ops()
total_gates = sum(op_counts.values())
depth = ansatz_isa.depth()
print("Operation counts:", op_counts)
print("Total number of gates:", total_gates)
print("Depth of the circuit: ", depth)

data_depth = {
    "Experiment": ["Hardware local penalty QAOA"],
    "Total number of gates": [total_gates],
    "Depth of the circuit": [depth]
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
    "Experiment": ["Classical Optimisation", "Quantum Optimisation (QAOA)", "Noisy Quantum Optimisation (Aer Simulator)", "Quantum Optimisation (QAOAAnsatz)"],
    "Ground State Energy": [eigenvalues[0], result.optimal_value + k, np.real(result1.best_measurement['value'] + k), np.real(result2.best_measurement['value'])],
    "Best Measurement": ["N/A", result.optimal_parameters, result1.best_measurement, result2.best_measurement],
    "Optimal Parameters": ["N/A", "N/A", "N/A", result2.optimal_parameters],
    "Execution Time (seconds)": [elapsed_time, elapsed_time, elapsed_time1, total_usage_time],
    "Total Gates": ["N/A", "N/A", total_gates, total_gates],
    "Circuit Depth": ["N/A", "N/A", depth, depth]
}

df.to_csv(file_path, index=False)