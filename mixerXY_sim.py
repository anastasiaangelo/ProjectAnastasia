# Point 4 of constraint studies for paper, Ising model with XY mixer to implement Hamming's condition

# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# %%
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os

num_res = 4
num_rot = 4
file_path = f"RESULTS/{num_rot}rot-XY-QAOA/{num_res}res-{num_rot}rot.csv"
# file_path = "RESULTS/hardware/7res-3rot-XY-hw.csv"
file_path_depth = f"RESULTS/Depths/{num_rot}rot-XY-QAOA-hw/{num_res}res-{num_rot}rot.csv"


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

k = 0
for i in range(num_qubits):
    k += 0.5 * q[i]

for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            k += 0.5 * 0.25 * Q[i][j]



# %% ############################################ Quantum optimisation ########################################################################
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
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

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)

def get_XY_mixer(num_qubits, num_rot, transverse_field=1):
    if num_rot < 2:
        raise ValueError("num_rot must be at least 2.")

    hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])

    for i in range(0, num_qubits - num_rot + 1, num_rot):  
        pauli_terms = {'X': [], 'Y': []}
        
        # Generate all pairwise (X, X) and (Y, Y) terms in each `num_rot` group
        for j in range(num_rot):
            for k in range(j + 1, num_rot):
                xx_term = ['I'] * num_qubits
                yy_term = ['I'] * num_qubits
                xx_term[i + j] = 'X'
                xx_term[i + k] = 'X'
                yy_term[i + j] = 'Y'
                yy_term[i + k] = 'Y'

                xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
                yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])

                hamiltonian += xx_op + yy_op

    hamiltonian *= transverse_field
    return -hamiltonian if num_rot == 2 else hamiltonian


def generate_initial_bitstring(num_qubits, num_rot):
    if num_rot < 2:
        raise ValueError("num_rot must be at least 2.")

    pattern = ['0'] * num_rot  
    pattern[0] = '1'  # Ensure at least one '1' per group
    bitstring = ''.join(pattern * ((num_qubits // num_rot) + 1))[:num_qubits]

    return bitstring

def create_product_state(base_circuit, n):
    num_qubits = base_circuit.num_qubits
    product_circuit = QuantumCircuit(num_qubits * n)

    for i in range(n):
        base_circuit_copy = deepcopy(base_circuit)
        product_circuit.compose(base_circuit_copy, qubits=range(i * num_qubits, (i + 1) * num_qubits), inplace=True)
        
    return product_circuit

def R_gate(theta=np.pi/4):
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(1)
    qc.ry(theta*np.pi/2, 0)
    qc.rz(np.pi, 0)
    op = Operator(qc)
    return op

def A_gate(theta=np.pi/4):
    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    rgate = R_gate(theta=theta)
    rgate_adj  = R_gate().adjoint()
    # apply rgate to qubit 0
    qc.unitary(rgate_adj, [1], label='R')
    qc.cx(0, 1)
    qc.unitary(rgate, [1], label='R')
    qc.cx(1, 0)
    return qc

def symmetry_preserving_initial_state(num_res, num_rot, theta=np.pi/4):
    if num_rot < 2:
        raise ValueError("num_rot must be at least 2.")

    qc = QuantumCircuit(num_rot)
    qc.x(num_rot // 2)
    agate = A_gate(theta=theta)

    for i in range(num_rot - 1):
        qc.compose(agate, [i, i + 1], inplace=True)

    init_state = create_product_state(qc, num_res)
    return init_state

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)


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

print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))

XY_mixer = get_XY_mixer(num_qubits, num_rot)
print('XY mixer: ', XY_mixer)

p = 1
mixer_bound = 1.0
cost_bound = 0.1
# generate a random vector initial_point of length 2*p, even indices should be drawn from a uniform distribution with bound cost_bound, odd indices should be drawn from a uniform distribution with bound mixer_bound
init_point_cost = np.random.uniform(-cost_bound, cost_bound, p)
init_point_mixer = np.random.uniform(-mixer_bound, mixer_bound, p)
initial_point = np.ones(2 * p)
initial_point[0::2] = init_point_cost
initial_point[1::2] = init_point_mixer

initial_bitstring = generate_initial_bitstring(num_qubits, num_rot)
state_vector = np.zeros(2**num_qubits)
indexx = int(initial_bitstring, 2)
state_vector[indexx] = 1
qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

# %% ############################################ Local simulation ########################################################################

start_time = time.time()
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point)
result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
end_time = time.time()

print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result.best_measurement)
elapsed_time = end_time - start_time
print(f"Local Simulation run time: {elapsed_time} seconds")
print('\n\n')

# with open(file_path, "a") as file:
#     file.write("\n\nThe result of the quantum optimisation using QAOA is: \n")
#     file.write(f"'best measurement' {result.best_measurement}\n")
#     file.write(f"Local Simulation run time: {elapsed_time} seconds\n")


# %% ############################################ Simulators ##########################################################################
from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import Sampler, BackendSampler
from qiskit.transpiler import PassManager

simulator = Aer.get_backend('qasm_simulator')
provider = IBMProvider()
available_backends = provider.backends()
print("Available Backends:", available_backends)
device_backend = provider.get_backend('ibm_brisbane')
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
intermediate_data = []
noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())

start_time1 = time.time()
qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point, callback=callback)
result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
opt_parameters = result1.optimal_parameters
print('Optimal parameters: ', opt_parameters)

# %% No post selection
# print("Best Measurement:", result1.best_measurement, flush=True)
# print('Ground state energy: ', np.real(result1.best_measurement['value'] + k), flush=True)
# data = {
#     "Experiment": ["Aer Simulation XY QAOA no Post Processing"],
#     "Ground State Energy": [np.real(result1.best_measurement['value'] + k)],
#     "Best Measurement": [result1.best_measurement],
#     "Execution Time (seconds)": [elapsed_time1],
#     "Number of qubits": [num_qubits],
#     "shots": [options['shots']]
# }

# df = pd.DataFrame(data)

# if not os.path.isfile(file_path):
#     # File does not exist, write with header
#     df.to_csv(file_path, index=False)
# else:
#     # File exists, append without writing the header
#     df.to_csv(file_path, mode='a', index=False, header=False)

# %% ############################################ Post Selection ##########################################################################
from qiskit_aer.primitives import Estimator
from qiskit import QuantumCircuit, transpile
import ast

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

def find_min_energy_and_bitstring_from_exact_energy_dataframe(df_exact, nres, nrot):
    df_filtered = df_exact[(df_exact['num_res'] == nres) & (df_exact['num_rot'] == nrot)]
    
    if len(df_filtered) > 1:
        raise Exception(f"Multiple rows found for num_res = {nres} and num_rot = {nrot}")
    
    energy = ast.literal_eval(df_filtered.iloc[0]['energies'])
    corresponding_bitstring = ast.literal_eval(df_filtered.iloc[0]['bitstrings'])
    
    energy = [complex(e).real for e in energy]

    if isinstance(energy, list):
        min_index = energy.index(min(energy))
        
        energy = energy[min_index]
        corresponding_bitstring = corresponding_bitstring[min_index]
    
    return energy, corresponding_bitstring


intermediate_data_dicts = []

for item in intermediate_data:
    for dict_item in item:
        intermediate_data_dicts.append(dict_item)

probability = []
total_arr = []
cumulative_probability_dict = {}
cumulative_total_dict = {}

exact_data = pd.read_csv("input_files/exact_energies_and_bitstrings.csv.gz", compression='gzip')
df_filtered = exact_data[(exact_data['num_res'] == num_res) & (exact_data['num_rot'] == num_rot)]
if df_filtered.empty: 
    raise Exception(f"No matching rows found for num_res = {num_res} and num_rot = {num_rot}")

# Instead of raising an error, select the row with the minimum energy
df_filtered = df_filtered.sort_values(by='energies').head(1)
min_energy, corresponding_bitstring = find_min_energy_and_bitstring_from_exact_energy_dataframe(exact_data, num_res, num_rot)

found_min_energy = False
first_iteration = None 

for i, dict in enumerate(intermediate_data_dicts):
    if found_min_energy:    
        break
    
    print(f"\n\nIteration {i+1}")
    print(f"Dictionary: {dict}")

    hits = 0.0
    total = 0.0

    for key in dict:
        bitstring = int_to_bitstring(key, num_qubits)
        energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
        print(f"Bitstring: {bitstring}, Energy: {energy}")

        if abs(energy - min_energy) < 1e-6:
            if first_iteration is None:  # Store only the first occurrence
                first_iteration = i+1
                print(f"Ground state {bitstring} first appeared at iteration {first_iteration} with energy {min_energy}")
            found_min_energy = True
            break

        if check_hamming(bitstring, num_rot):
            hits += dict[key]
            total += dict[key]
            #print(f"Bitstring: {bitstring} has a value of {dict[key]}")
            if bitstring in cumulative_probability_dict:
                cumulative_probability_dict[bitstring] += dict[key]
            else:
                cumulative_probability_dict[bitstring] = dict[key]
        else:
            total += dict[key]
        if bitstring in cumulative_total_dict:
            cumulative_total_dict[bitstring] += dict[key]
        else:
            cumulative_total_dict[bitstring] = dict[key]
            #print(f"Bitstring: {bitstring} does not satisfy the Hamming condition.")
            #pass
    
    probability.append(hits)
    total_arr.append(total)


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

                all_bitstrings[intermediate_bitstring]['probability'] += probability  
                energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)
                count = all_bitstrings[intermediate_bitstring]['count']
                all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * count + energy) / (count + 1)
                all_bitstrings[intermediate_bitstring]['count'] += 1

    


total_probabilities = sum(bitstring_data['probability'] for bitstring_data in all_bitstrings.values())
for bitstring_data in all_bitstrings.values():
    bitstring_data['probability'] /= total_probabilities

sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])

total_bitstrings = sum(
    probability * options['shots'] 
    for data in intermediate_data
    for distribution in data['quasi_distributions']
    for int_bitstring, probability in distribution.items()
) + sum(probability * options['shots'] for state, probability in final_bitstrings.items()
)

hamming_satisfying_bitstrings = sum(bitstring_data['probability'] * options['shots'] for bitstring_data in all_bitstrings.values())
fraction_satisfying_hamming = hamming_satisfying_bitstrings / total_bitstrings
print(f"Fraction of bitstrings that satisfy the Hamming constraint: {fraction_satisfying_hamming}")

# ground_state_repetition = sorted_bitstrings[0][1]['index']

print("Best Measurement:", best_measurement, flush=True)
for bitstring, data in sorted_bitstrings:
    print(f"Bitstring: {bitstring}, Probability: {data['probability']}, Energy: {data['energy']}", flush=True)

found = False
for bitstring, data in sorted_bitstrings:
    if bitstring == best_measurement['bitstring']:
        print('Best measurement bitstring respects Hammings conditions.\n', flush=True)
        print('Ground state energy: ', data['energy']+k, flush=True)
        data = {
            "Experiment": ["Aer Simulation XY QAOA"],
            "Ground State Energy": [np.real(result1.best_measurement['value'] + k)],
            "Best Measurement": [result1.best_measurement],
            "Execution Time (seconds)": [elapsed_time1],
            "Number of qubits": [num_qubits],
            "shots": [options['shots']],
            "Fraction": [fraction_satisfying_hamming],
            # "Iteration Ground State": [ground_state_repetition],
            "Sorted Bitstrings": [sorted_bitstrings]
}
        found = True
        break

if not found:
    print('Best measurement bitstring does not respect Hammings conditions, take the sorted bitstring corresponding to the smallest energy.\n', flush=True)
    post_selected_bitstring, post_selected_energy = sorted_bitstrings[0]
    data = {
        "Experiment": ["Aer Simulation XY QAOA, post-selected"],
        "Ground State Energy": [post_selected_energy['energy'] + k],
        "Best Measurement": [post_selected_bitstring],
        "Execution Time (seconds)": [elapsed_time1],
        "Number of qubits": [num_qubits],
        "shots": [options['shots']],
        "Fraction": [fraction_satisfying_hamming],
        # "Iteration Ground State": [ground_state_repetition],
        "Sorted Bitstrings": [sorted_bitstrings]

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

ansatz = QAOAAnsatz(q_hamiltonian, mixer_operator=XY_mixer, reps=p)
print('\n\nQAOAAnsatz: ', ansatz)

# To pass optimial parameters from the local simualtion to the hardware QAOAAnsatz
opt_parameters = [3.15625, 1.0]
ansatz_one_rep = QAOAAnsatz(q_hamiltonian, mixer_operator=XY_mixer, reps=1)
params_dict = {param: value for param, value in zip(ansatz_one_rep.parameters, opt_parameters)}
bound_circuit = ansatz_one_rep.assign_parameters(params_dict)

print("Parameters in the new QAOA ansatz with one repetition:")
for param, value in params_dict.items():
    print(f"{param}: {value}")


target = backend.target

def callback(quasi_dists, parameters, energy):
    intermediate_data_hw.append({
        'quasi_distributions': quasi_dists,
        'parameters': parameters,
        'energy': energy
    })

intermediate_data_hw = []


# %%
def generate_linear_coupling_map(num_qubits):
    coupling_list = [[i, i + 1] for i in range(num_qubits - 1)]
    return CouplingMap(couplinglist=coupling_list)

linear_coupling_map = generate_linear_coupling_map(num_qubits)
# coupling_map = CouplingMap(couplinglist=[[0, 1],[0, 15], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14], [14, 13], [15, 0], [16, 4], [17, 8]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [17, 8], [18, 12], [19, 15]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24]])
coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [25, 35], [26, 25], [26, 27], [27, 17], [27, 26]])
# coupling_map = CouplingMap(couplinglist=[[0, 1], [0, 15], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [4, 16], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [8, 17], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [12, 18], [13, 12], [13, 14], [14, 13], [15, 0], [15, 19], [16, 4], [16, 23], [17, 8], [17, 27], [18, 12], [19, 15], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [22, 21], [22, 23], [23, 16], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [26, 25], [26, 27], [27, 17], [27, 26], [27, 28], [28, 27], [28, 29], [29, 28]])

qr = QuantumRegister(num_qubits, 'q')
circuit = QuantumCircuit(qr)
trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=coupling_map,
                       optimization_level=3, layout_method='trivial', routing_method='stochastic')
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
qaoa2 = SamplingVQE(sampler=sampler, ansatz=ansatz_isa, optimizer=COBYLA(), initial_point=initial_point, callback=callback)
print('here 3')
result2 = qaoa2.compute_minimum_eigenvalue(hamiltonian_isa)

# shots = 10000 
# sampler_one_rep = Sampler(backend=backend, session=session, options={'shots': shots})
# qaoa_one_rep = SamplingVQE(sampler=sampler_one_rep, ansatz=ansatz_one_rep_isa, optimizer=COBYLA(), initial_point=initial_point, callback=callback)
# result_one_rep = qaoa_one_rep.compute_minimum_eigenvalue(hamiltonian_isa_one_rep)

print("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
print('best measurement', result2.best_measurement)
print('Optimal parameters: ', result2.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result2.best_measurement['value']))

# print("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
# print('best measurement', result_one_rep.best_measurement)
# print('Optimal parameters: ', result_one_rep.optimal_parameters)
# print('The ground state energy with noisy QAOA is: ', np.real(result_one_rep.best_measurement['value']))

# %%
jobs = service.jobs(session_id='crrdap27jqmg008z9m00')

for job in jobs:
    if job.status().name == 'DONE':
        results = job.result()
        print("Job completed successfully")
else:
    print("Job status:", job.status())

# %%
from qiskit.quantum_info import Statevector, Operator

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

def get_best_measurement_from_sampler_result(sampler_result, q_hamiltonian, num_qubits):
    if not hasattr(sampler_result, 'quasi_dists') or not isinstance(sampler_result.quasi_dists, list):
        raise ValueError("SamplerResult does not contain 'quasi_dists' as a list")

    best_bitstring = None
    lowest_energy = float('inf')
    highest_probability = -1

    for quasi_distribution in sampler_result.quasi_dists:
        for int_bitstring, probability in quasi_distribution.items():
            bitstring = format(int_bitstring, '0{}b'.format(num_qubits))  # Ensure bitstring is string
            energy = evaluate_energy(bitstring, q_hamiltonian)
            if energy < lowest_energy:
                lowest_energy = energy
                best_bitstring = bitstring
                highest_probability = probability

    return best_bitstring, highest_probability, lowest_energy


best_bitstring, probability, value = get_best_measurement_from_sampler_result(results, q_hamiltonian, num_qubits)
print(f"Best measurement: {best_bitstring} with ground state energy {value} and probability {probability}")

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
            logical_bitstring = bitstring[:logical_qubits]
            total_bitstrings += 1
            if check_hamming(logical_bitstring, num_rot):
                energy = calculate_bitstring_energy(logical_bitstring, q_hamiltonian, backend)
                print(f"Bitstring: {logical_bitstring}, Energy: {energy}, Probability: {probability}")
                valid_bitstrings += 1
                if energy < lowest_energy:
                    lowest_energy = energy
                    best_bitstring = logical_bitstring
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

eigenstate_distribution_hw = result2.eigenstate
best_measurement_hw = result2.best_measurement
final_bitstrings_hw = {state: probability for state, probability in eigenstate_distribution_hw.items()}

all_bitstrings_hw = {}
for state, prob in final_bitstrings_hw.items():
    bitstring_hw = int_to_bitstring(state, num_qubits)
    if check_hamming(bitstring_hw, num_rot):
        if bitstring_hw not in all_bitstrings_hw:
            all_bitstrings_hw[bitstring_hw] = {'probability': 0, 'energy': 0, 'count': 0}
        all_bitstrings_hw[bitstring_hw]['probability'] += prob  # Aggregate probabilities
        energy_hw = calculate_bitstring_energy(bitstring_hw, q_hamiltonian)
        all_bitstrings_hw[bitstring_hw]['energy'] = (all_bitstrings_hw[bitstring_hw]['energy'] * all_bitstrings_hw[bitstring_hw]['count'] + energy_hw) / (all_bitstrings_hw[bitstring_hw]['count'] + 1)
        all_bitstrings_hw[bitstring_hw]['count'] += 1

for data_hw in intermediate_data_hw:
    print(f"Quasi Distribution: {data_hw['quasi_distributions']}, Parameters: {data_hw['parameters']}, Energy: {data_hw['energy']}", flush=True)
    for distribution_hw in data_hw['quasi_distributions']:
        for int_bitstring, probability in distribution_hw.items():
            intermediate_bitstring_hw = int_to_bitstring(int_bitstring, num_qubits)
            if check_hamming(intermediate_bitstring_hw, num_rot):
                if intermediate_bitstring_hw not in all_bitstrings_hw:
                    all_bitstrings_hw[intermediate_bitstring_hw] = {'probability': 0, 'energy': 0, 'count': 0}
                all_bitstrings_hw[intermediate_bitstring_hw]['probability'] += probability  # Aggregate probabilities
                energy_hw = calculate_bitstring_energy(intermediate_bitstring_hw, q_hamiltonian)
                all_bitstrings_hw[intermediate_bitstring_hw]['energy'] = (all_bitstrings_hw[intermediate_bitstring_hw]['energy'] * all_bitstrings_hw[intermediate_bitstring_hw]['count'] + energy_hw) / (all_bitstrings_hw[intermediate_bitstring_hw]['count'] + 1)
                all_bitstrings_hw[intermediate_bitstring_hw]['count'] += 1


sorted_bitstrings_hw = sorted(all_bitstrings_hw.items(), key=lambda x: x[1]['energy'])

print("Best Measurement:", best_measurement_hw, flush=True)
for bitstring, data in sorted_bitstrings_hw:
    print(f"Bitstring: {bitstring_hw}, Probability: {data_hw['probability']}, Energy: {data_hw['energy']}", flush=True)

found = False
for bitstring_hw, data_hw in sorted_bitstrings_hw:
    if bitstring_hw == best_measurement_hw['bitstring']:
        print('Best measurement bitstring respects Hammings conditions.\n', flush=True)
        print('Ground state energy: ', data_hw['energy']+k, flush=True)
        data_hw = {
            "Experiment": ["Aer Simulation XY QAOA"],
            "Ground State Energy": [np.real(result2.best_measurement['value'] + k)],
            "Best Measurement": [result2.best_measurement],
            "Execution Time (seconds)": [total_usage_time],
            "Number of qubits": [num_qubits],
            "shots": [options['shots']]
}
        found = True
        break

if not found:
    print('Best measurement bitstring does not respect Hammings conditions, take the sorted bitstring corresponding to the smallest energy.\n', flush=True)
    post_selected_bitstring, post_selected_energy = sorted_bitstrings_hw[0]
    data_hw = {
        "Experiment": ["Aer Simulation XY QAOA, post-selected"],
        "Ground State Energy": [post_selected_energy['energy'] + k],
        "Best Measurement": [post_selected_bitstring],
        "Execution Time (seconds)": [elapsed_time1],
        "Number of qubits": [num_qubits],
        "shots": [options['shots']]
    }

df = pd.DataFrame(data_hw)

if not os.path.isfile(file_path):
    # File does not exist, write with header
    df.to_csv(file_path, index=False)
else:
    # File exists, append without writing the header
    df.to_csv(file_path, mode='a', index=False, header=False)


with open(file_path, "a") as file:
    file.write("\n\nThe result of the noisy quantum optimisation using QAOAAnsatz is: \n")
    file.write(f"'best measurement' {result2.best_measurement}")
    file.write(f"Optimal parameters: {result2.optimal_parameters}")
    file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result2.best_measurement['value'])}")
    file.write(f"Total Usage Time Hardware: {total_usage_time} seconds")
    file.write(f"Total number of gates: {total_gates}\n")
    file.write(f"Depth of circuit: {depth}\n")

# %%
index = ansatz_isa.layout.final_index_layout() # Maps logical qubit index to its position in bitstring

# measured_bitstring = result2.best_measurement['bitstring']
measured_bitstring = best_bitstring
original_bitstring = ['']*num_qubits

for i, logical in enumerate(index):
        original_bitstring[i] = measured_bitstring[logical]

original_bitstring = ''.join(original_bitstring)
print("Original bitstring:", original_bitstring)

data = {
    "Experiment": ["Classical Optimisation", "Quantum Optimisation (QAOA)", "Noisy Quantum Optimisation (Aer Simulator)", "Quantum Optimisation (QAOAAnsatz)"],
    "Ground State Energy": ["N/A", result.optimal_value + k, np.real(result1.best_measurement['value'] + k), np.real(result2.best_measurement['value'])],
    "Best Measurement": ["N/A", result.optimal_parameters, result1.best_measurement, result2.best_measurement],
    "Optimal Parameters": ["N/A", "N/A", "N/A", result2.optimal_parameters],
    "Execution Time (seconds)": [elapsed_time, elapsed_time, elapsed_time1, total_usage_time],
    "Total Gates": ["N/A", "N/A", total_gates, total_gates],
    "Circuit Depth": ["N/A", "N/A", depth, depth]
}

df.to_csv(file_path, index=False)