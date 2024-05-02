# Point 5 of constraint studies, creation and annihilation operators for multiple rotamers per residue, no constraints required

# %%
import numpy as np
import pandas as pd
import itertools
import functools
import operator
import time
from itertools import combinations
from qiskit.visualization import plot_histogram

file_path = "RESULTS/creation-annihilation-QAOA/2res-4rot"

qubit_per_res = 1
num_rot = 2**qubit_per_res

df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("energy_files/two_body_terms.csv")
v = df['E_ij'].values
numm = len(v)

print("q: \n", q)

num_qubits = N_res * qubit_per_res

# %%  Quantum optimisation
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

## Mapping to qubits
H_self = SparsePauliOp(Pauli('I'* num_qubits), coeffs=[0])
H_int = SparsePauliOp(Pauli('I'* num_qubits), coeffs=[0]) 

def N_0(i, n):
    pauli_str = ['I'] * n
    pauli_str[i] = 'Z'
    z_op = SparsePauliOp(Pauli(''.join(pauli_str)), coeffs=[0.5])
    i_op = SparsePauliOp(Pauli('I'*n), coeffs=[0.5])
    return z_op + i_op

def N_1(i, n):
    pauli_str = ['I'] * n
    pauli_str[i] = 'Z'
    z_op = SparsePauliOp(Pauli(''.join(pauli_str)), coeffs=[-0.5])
    i_op = SparsePauliOp(Pauli('I'*n), coeffs=[0.5])
    return z_op + i_op


def create_pauli_operators(num_qubits, qubits_per_res):
    operators = []
    for i in range(0, num_qubits, qubits_per_res):
        # Generate all combinations of N_0 and N_1 for the residue
        for comb in itertools.product([N_0, N_1], repeat=qubits_per_res):
            ops = [func(j, num_qubits) for j, func in enumerate(comb, start=i)]
            # Now you have a list of operators for each qubit in the residue
            full_op = functools.reduce(operator.matmul, ops)
            operators.append(full_op)
    return operators

for i, op in enumerate(create_pauli_operators(num_qubits, qubit_per_res)):
    H_self += q[i] * op
    if i >= len(q) - 1:
        break

def create_interaction_operators(num_qubits, qubits_per_res, v):
    H_int = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
    v_index = 0
    
    # Iterate over all unique pairs of residues
    for res1, res2 in combinations(range(0, num_qubits, qubits_per_res), 2):
        # Generate all combinations of N_0 and N_1 for each qubit in the residue
        for comb1 in itertools.product([N_0, N_1], repeat=qubits_per_res):
            for comb2 in itertools.product([N_0, N_1], repeat=qubits_per_res):
                op_list1 = [func(i + res1, num_qubits) for i, func in enumerate(comb1)]
                op_list2 = [func(j + res2, num_qubits) for j, func in enumerate(comb2)]
                
                # Now you have two lists of operators, one for each residue in the pair
                full_op1 = functools.reduce(operator.matmul, op_list1)
                full_op2 = functools.reduce(operator.matmul, op_list2)
                
                H_int += v[v_index] * full_op1 @ full_op2
                v_index += 1
                
                if v_index >= len(v) - 1:
                    break
            if v_index >= len(v) - 1:
                break
        if v_index >= len(v) - 1:
            break

    return H_int

H_int = create_interaction_operators(num_qubits, qubit_per_res, v)


H_gen = H_int + H_self

def X_op(i, num):
    op_list = ['I'] * num
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))

mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
p = 1
initial_point = np.ones(2*p)
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result_gen = qaoa.compute_minimum_eigenvalue(H_gen)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result_gen.best_measurement)
print('The ground state energy with QAOA is: ', np.real(result_gen.best_measurement['value']))

with open(file_path, "w") as file:
    file.write("\n\nLocal Quantum optimisation results.\n")
    file.write(f"'best measurement' {result_gen.best_measurement}\n")
    file.write(f"The ground state energy with QAOA is: ' {np.real(result_gen.best_measurement['value'])}\n")

# %% ############################################ Simulators ##########################################################################
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
result1 = qaoa1.compute_minimum_eigenvalue(H_int)
end_time1 = time.time()

print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
print('best measurement', result1.best_measurement)
print('Optimal parameters: ', result1.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
elapsed_time1 = end_time1 - start_time1
print(f"Aer Simulator run time: {elapsed_time1} seconds")
print('\n\n')

with open(file_path, "a") as file:
    file.write("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
    file.write(f"'best measurement' {result1.best_measurement}")
    file.write(f"Optimal parameters: {result1.optimal_parameters}")
    file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result1.best_measurement['value'])}")
    file.write(f"Aer Simulator run time: {elapsed_time1} seconds")

# %% ############################################# Hardware with QAOAAnastz ##################################################################
from qiskit_ibm_provider import IBMProvider
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.transpiler import CouplingMap, Layout
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_algorithms import SamplingVQE

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
backend.configuration().default_rep_delay == 0.00001 #to speed up execution with dynamic repetition rate
print('Coupling Map of hardware: ', backend.configuration().coupling_map)

ansatz = QAOAAnsatz(H_int, mixer_operator=mixer_op, reps=p)
print('\n\nQAOAAnsatz: ', ansatz)

target = backend.target

def generate_linear_coupling_map(num_qubits):

    coupling_list = [[i, i + 1] for i in range(num_qubits - 1)]
    
    return CouplingMap(couplinglist=coupling_list)

linear_coupling_map = generate_linear_coupling_map(num_qubits)
# coupling_map = CouplingMap(couplinglist=[[0, 1],[0, 15], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]])
qr = QuantumRegister(num_qubits, 'q')
circuit = QuantumCircuit(qr)
trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
ansatz_isa = transpile(ansatz, backend=backend, initial_layout=trivial_layout, coupling_map=linear_coupling_map,
                       optimization_level= 3, layout_method='dense', routing_method='stochastic')
print("\n\nAnsatz layout after explicit transpilation:", ansatz_isa._layout)

hamiltonian_isa = H_int.apply_layout(ansatz_isa.layout)
print("\n\nAnsatz layout after transpilation:", hamiltonian_isa)

# %%
ansatz_isa.decompose().draw('mpl')

op_counts = ansatz_isa.count_ops()
total_gates = sum(op_counts.values())
depth = ansatz_isa.depth()
print("Operation counts:", op_counts)
print("Total number of gates:", total_gates)
print("Depth of the circuit: ", depth)

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
print('The ground state energy with noisy QAOA is: ', np.real(result2.best_measurement['value']))

# %%
jobs = service.jobs(session_id='crqsrk22x9jg008z2qe0')
for job in jobs:
    if job.status().name == 'DONE':
        results = job.result()
        print("Job completed successfully")
else:
    print("Job status:", job.status())

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


best_bitstring, probability, value = get_best_measurement_from_sampler_result(results, H_int, num_qubits)
print(f"Best measurement: {best_bitstring} with ground state energy {value} and probability {probability}")

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
    file.write(f"'The ground state energy with noisy QAOA is: ' {np.real(result2.best_measurement['value'])}")
    file.write(f"Total Usage Time Hardware: {total_usage_time} seconds")
    file.write(f"Total number of gates: {total_gates}\n")   
    file.write(f"Depth of circuit: {depth}\n")
