import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import functools
import operator


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
from qiskit import Aer, QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

## Mapping to qubits
# for 2 qubits per residue, 4 rotamers per residue
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

def generate_n_operators(start_qubit, qubit_per_res, num_qubits):
    n_operators = []
    for qubit in range(start_qubit, start_qubit + qubit_per_res):
        n_operators.append(N_0(qubit, num_qubits))
        n_operators.append(N_1(qubit, num_qubits))
    return n_operators

i = 0 #each loop is one residue
for j in range(0, num_qubits, qubit_per_res):
    n_ops = generate_n_operators(j, qubit_per_res, num_qubits)
    print('operators: ', n_ops)
    for comb in itertools.product(*[n_ops]):
        H_self += q[i] * functools.reduce(operator.matmul, comb)
        print('hamiltoanian: ', H_self)
        i += 1
        if i >= num:
            break


i = 0     #one loop is one pair of residues
for j in range(0, num_qubits, qubit_per_res):
    for k in range(j + qubit_per_res, num_qubits, qubit_per_res):
        n_ops_j = generate_n_operators(j, qubit_per_res, num_qubits)
        n_ops_k = generate_n_operators(k, qubit_per_res, num_qubits)
        for comb in itertools.product(*[n_ops_j, n_ops_k]):
            H_int += v[i] * functools.reduce(operator.matmul, comb)
            i += 1
            if i >= numm:
                break
    

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
print(result_gen)

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeKolkata, FakeVigo
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session, Sampler

IBMProvider.save_account('25a4f69c2395dfbc9990a6261b523fe99e820aa498647f92552992afb1bd6b0bbfcada97ec31a81a221c16be85104beb653845e23eeac2fe4c0cb435ec7fc6b4', overwrite=True)
provider = IBMProvider()
available_backends = provider.backends()
print([backend.name for backend in available_backends])
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibmq_qasm_simulator")
noise_model = NoiseModel.from_backend(backend)
simulator = AerSimulator(noise_model = noise_model)
fake_backend = FakeKolkata()
noise_model = NoiseModel.from_backend(fake_backend)
options = Options()
options.simulator = {
    "noise_model": noise_model,
    "basis_gates": fake_backend.configuration().basis_gates,
    "coupling_map": fake_backend.configuration().coupling_map,
    "seed_simulator": 42
}
options.execution.shots = 1000
options.optimization_level = 0
options.resilience_level = 0

with Session(service=service, backend=backend):
    sampler = Sampler(options=options)
    qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
    result1 = qaoa1.compute_minimum_eigenvalue(H_gen)

print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
print('best measurement', result1.best_measurement)
print('Optimal parameters: ', result1.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
print(result1)



