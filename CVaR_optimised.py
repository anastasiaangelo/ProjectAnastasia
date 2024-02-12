
import numpy as np
import pandas as pd
import itertools
import functools
import operator
from itertools import combinations
from qiskit.visualization import plot_histogram


qubit_per_res = 2
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

## Quantum optimisation
from qiskit import Aer
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes

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
        for comb in itertools.product([N_0, N_1], repeat=qubits_per_res):
            ops = [func(j, num_qubits) for j, func in enumerate(comb, start=i)]
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
    
    for res1, res2 in combinations(range(0, num_qubits, qubits_per_res), 2):
        for comb1 in itertools.product([N_0, N_1], repeat=qubits_per_res):
            for comb2 in itertools.product([N_0, N_1], repeat=qubits_per_res):
                op_list1 = [func(i + res1, num_qubits) for i, func in enumerate(comb1)]
                op_list2 = [func(j + res2, num_qubits) for j, func in enumerate(comb2)]
                
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
print("\n\nThe result of the quantum optimisation using QAOA without CVar is: \n")
print('best measurement', result_gen.best_measurement)
print('The ground state energy with QAOA is: ', np.real(result_gen.best_measurement['value']))


counts = result_gen.best_measurement
histogram = plot_histogram(counts, title="QAOA Measurement Results")
histogram.savefig('qaoa_measurement_results.jpg', format='jpg')


from qiskit_aer.noise import NoiseModel, QuantumError, pauli_error
from qiskit.quantum_info import Kraus
from qiskit_aer.primitives import Sampler
from qiskit.primitives import Sampler, BackendSampler
from qiskit.transpiler import PassManager

backend = Aer.get_backend('qasm_simulator')
noise_model = NoiseModel()
prob_x = 0.05  # Probability for X error
prob_sx = 0.02  # Probability for SX error

# Create quantum errors
error_ops = [np.sqrt(1 - prob_sx) * np.eye(2), np.sqrt(prob_sx) * Pauli('X').to_matrix()]

error_x = QuantumError(pauli_error([('X', prob_x), ('I', 1 - prob_x)]))
# error_sx = QuantumError(pauli_error([('SX', prob_sx), ('I', 1 - prob_sx)]))
error_sx = QuantumError(Kraus(error_ops))

# Create a new noise model
new_noise_model = NoiseModel()

# Add quantum errors to the noise model for specific gates
new_noise_model.add_quantum_error(error_x, 'x', [0])  # Apply to qubit 0
new_noise_model.add_quantum_error(error_sx, 'sx', [1])

options= {
    "noise_model": new_noise_model,
    "basis_gates": backend.configuration().basis_gates,
    "coupling_map": backend.configuration().coupling_map,
    "seed_simulator": 42,
    "shots": 1000,
    "optimization_level": 0,
    "resilience_level": 0
}

noisy_sampler = BackendSampler(backend=backend, options=options, bound_pass_manager=PassManager())

maxiter = 100
n = 4
optimizer = COBYLA(maxiter=maxiter)
ansatz = RealAmplitudes(n, reps=1)
m = ansatz.num_parameters
sampler = Sampler()
initial_point = np.ones(2*p)
alphas = [1.0, 0.50, 0.25, 0.125, 0.05, 0.025, 0.005]

objectives = {alpha: [] for alpha in alphas}
results = {}

print("\n\nThe result of the CVaR QAOA Optimisation is: \n")

for alpha in alphas:

    qaoaC = QAOA(
        sampler=noisy_sampler,
        optimizer=optimizer,
        reps=p, 
        mixer=mixer_op,
        initial_point= initial_point,
        aggregation=alpha
    )

    results[alpha] = qaoaC.compute_minimum_eigenvalue(H_gen)

    for alpha, result in results.items():
        eigenvalue = result.eigenvalue.real
    
        print("alpha = {}:".format(alpha))
        print("Minimum Eigenvalue: ", eigenvalue)
        print("Bitstring: ", result.best_measurement['bitstring'])
        print()

qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result1 = qaoa1.compute_minimum_eigenvalue(H_gen)

print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
print('best measurement', result1.best_measurement)
print('Optimal parameters: ', result1.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
