import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
import cmath

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

## Classical optimisation:
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig


## Quantum optimisation
from qiskit import Aer, QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

## Creation and Annihilation operators Hamiltonian
num_qubits = N_res * qubit_per_res

## Classically

H_s = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
H_i = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

a = 0.5*(X + 1j*Y)
a_dagger = 0.5 *(X - 1j*Y) 

identity = np.eye(2)

aad = a@a_dagger
ada = a_dagger@a

def extended_operator(n, qubit, op):
    ops = [identity if i != qubit else op for i in range(n)]
    extended_op = ops[0]
    for op in ops[1:]:
        extended_op = np.kron(extended_op, op)
    return extended_op


s = 0        
for i in range(0, num_qubits, qubit_per_res):
    aad_extended = extended_operator(num_qubits, i, aad)
    ada_extended = extended_operator(num_qubits, i, ada)
    aad_extended1 = extended_operator(num_qubits, i+1, aad)
    ada_extended1 = extended_operator(num_qubits, i+1, ada)
    H_s += q[s] * aad_extended @ aad_extended1 + q[s+1] * aad_extended @ ada_extended1 + q[s+2] * ada_extended @ aad_extended1 + q[s+3] * ada_extended @ ada_extended1
    s += num_rot
    if s >= num:
        break

k = 0
for i in range(0, num_qubits, qubit_per_res):
    for j in range(qubit_per_res, num_qubits, qubit_per_res):
        aad_extended = extended_operator(num_qubits, i, aad)
        ada_extended = extended_operator(num_qubits, i, ada)
        aad_extended1 = extended_operator(num_qubits, i+1, aad)
        ada_extended1 = extended_operator(num_qubits, i+1, ada)
        aad_extendedj = extended_operator(num_qubits, j, aad)
        ada_extendedj = extended_operator(num_qubits, j, ada)
        aad_extendedj1 = extended_operator(num_qubits, j+1, aad)
        ada_extendedj1 = extended_operator(num_qubits, j+1, ada)
        if k >= numm:
            break
        H_i += v[k] * aad_extended @ aad_extended1 @ aad_extendedj @ aad_extendedj1 + v[k+1] * aad_extended @ aad_extended1 @ aad_extendedj @ ada_extendedj1 + v[k+2] * aad_extended @ aad_extended1 @ ada_extendedj @ aad_extendedj1 + v[k+3] * aad_extended @ aad_extended1 @ ada_extendedj @ ada_extendedj1 + \
                v[k+4] * aad_extended @ ada_extended1 @ aad_extendedj @ aad_extendedj1 + v[k+5] * aad_extended @ ada_extended1 @ ada_extendedj @ ada_extendedj1 + v[k+6] * aad_extended @ ada_extended1 @ ada_extendedj @ aad_extendedj1 + v[k+7] * aad_extended @ ada_extended1 @ ada_extendedj @ ada_extendedj1 + \
                v[k+8] * ada_extended @ aad_extended1 @ aad_extendedj @ aad_extendedj1 + v[k+9] * ada_extended @ aad_extended1 @ aad_extendedj @ ada_extendedj1 + v[k+10] * ada_extended @ aad_extended1 @ ada_extendedj @ aad_extendedj1 + v[k+11] * ada_extended @ aad_extended1 @ ada_extendedj @ ada_extendedj1 + \
                v[k+12] * ada_extended @ ada_extended1 @ aad_extendedj @ aad_extendedj1 + v[k+13] * ada_extended @ ada_extended1 @ aad_extendedj @ ada_extendedj1 + v[k+14] * ada_extended @ ada_extended1 @ ada_extendedj @ aad_extendedj1 + v[k+15] * ada_extended @ ada_extended1 @ ada_extendedj @ ada_extendedj1

        k += num_rot**2
    
H_tt = H_i + H_s 

eigenvalue, eigenvector = eigsh(H_tt, k=num_qubits, which='SA')
print('\nThe ground state with the number operator classically is: ', eigenvalue[0])

ground_state= eig(H_tt)
print('eig result:', ground_state)


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


i = 0 #each loop is one residue
for j in range(0, num_qubits, qubit_per_res):
    N_0i = N_0(j, num_qubits)
    N_1i = N_1(j, num_qubits)
    N_0j = N_0(j+1, num_qubits)
    N_1j = N_1(j+1, num_qubits)
    H_self += q[i] * N_0i @ N_0j + q[i+1] * N_0i @ N_1j + q[i+2] * N_1i @ N_0j + q[i+3] * N_1i @ N_1j 
    i += num_rot
    if i >= num:
        break


i = 0     #one loop is one pair of residues
for j in range(0, num_qubits, qubit_per_res):
    for k in range(qubit_per_res, num_qubits, qubit_per_res):
        N_0i = N_0(j, num_qubits)
        N_1i = N_1(j, num_qubits)
        N_0ii = N_0(j+1, num_qubits)
        N_1ii = N_1(j+1, num_qubits)
        N_0j = N_0(k, num_qubits)
        N_1j = N_1(k, num_qubits)
        N_0jj = N_0(k+1, num_qubits)
        N_1jj = N_1(k+1, num_qubits)
        if i >= numm:
            break
        H_int += v[i] * N_0i @ N_0ii @ N_0j @ N_0jj + v[i+1] * N_0i @ N_0ii @ N_0j @ N_1jj + v[i+2] * N_0i @ N_0ii @ N_1j @ N_0jj + v[i+3] * N_0i @ N_0ii @ N_1j @ N_1jj + \
                + v[i+4] * N_0i @ N_1ii @ N_0j @ N_0jj + v[i+5] * N_0i @ N_1ii @ N_1j @ N_1jj + v[i+6] * N_0i @ N_1ii @ N_1j @ N_0jj + + v[i+7] * N_0i @ N_1ii @ N_1j @ N_1jj + \
                + v[i+8] * N_1i @ N_0ii @ N_0j @ N_0jj + v[i+9] * N_1i @ N_0ii @ N_0j @ N_1jj + v[i+10] * N_1i @ N_0ii @ N_1j @ N_0jj + v[i+11] * N_1i @ N_0ii @ N_1j @ N_1jj + \
                + v[i+12] * N_1i @ N_1ii @ N_0j @ N_0jj + v[i+13] * N_1i @ N_1ii @ N_0j @ N_1jj + v[i+14] * N_1i @ N_1ii @ N_1j @ N_0jj + v[i+15] * N_1i @ N_1ii @ N_1j @ N_1jj 
        
        i += num_rot**2
    

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
    print('Running noisy simulation..')

print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
print('best measurement', result1.best_measurement)
print('Optimal parameters: ', result1.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
print(result1)


