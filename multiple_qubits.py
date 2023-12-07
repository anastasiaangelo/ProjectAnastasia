import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
import cmath

qubit_per_res = 2
num_rot = 2**qubit_per_res


df1 = pd.read_csv("one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("two_body_terms.csv")
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

## First Classically

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


# bitstring = result.best_measurement['bitstring']

# for i in range(num):
#     Q[i][i] = deepcopy(q[i])

# E_0 = np.zeros(num_qubits)
# # E_0 = np.zeros(num-N_res)
# E_1 = np.zeros(num_qubits)

# t_0 = 0
# t_1 = 0

# for i in range(num):
#     if bitstring[i] == '0' and t_0 < num_qubits:
#     # if bitstring[i] == '0' and t_0 < num-N_res:
#         E_0[t_0] = Q[i][i]
#         t_0 += 1
#     elif bitstring[i] == '1' and t_1 < num_qubits:
#         E_1[t_1] = Q[i][i]
#         t_1 += 1

# E_00 = np.zeros((num_qubits, num_qubits))
# E_01 = np.zeros((num_qubits, num_qubits))
# E_10 = np.zeros((num_qubits, num_qubits))
# E_11 = np.zeros((num_qubits, num_qubits))

# t_00 = 0
# t_01 = 0
# t_10 = 0
# t_11 = 0

# for i in range(num-num_rot):

#     if bitstring[i] == '0' and t_00 < N_res:
#         if i % 2 != 0:
#             for j in range(i+1, i+num_rot+1):
#                 if bitstring[j] == '0':
#                     E_00[t_00][t_00+1] = Q[i][j]
#                     t_00 += 1
#         else:
#             for j in range(i+num_rot, i+num_rot+2):
#                 if bitstring[j] == '0':
#                     E_00[t_00][t_00+1] = Q[i][j]
#                     t_00 += 1

#     if bitstring[i] == '0' and t_01 < N_res:
#         if i % 2 != 0:
#             for j in range(i+1, i+num_rot+1):
#                 if bitstring[j] == '1':
#                     E_01[t_01][t_01+1] = Q[i][j]
#                     t_01 += 1
#         else:
#             for j in range(i+num_rot, i+num_rot+2):
#                 if bitstring[j] == '1':
#                     E_01[t_01][t_01+1] = Q[i][j]
#                     t_01 += 1

#     if bitstring[i] == '1' and t_10 < N_res:
#         if i % 2 != 0:
#             for j in range(i+1, i+num_rot+1):
#                 if bitstring[j] == '0':
#                     E_10[t_10][t_10+1] = Q[i][j]
#                     t_10 += 1
#         else:
#             for j in range(i+num_rot, i+num_rot+2):
#                 if bitstring[j] == '0':
#                     E_10[t_10][t_10+1] = Q[i][j]
#                     t_10 += 1

#     if bitstring[i] == '1' and t_11 < N_res:
#         if i % 2 != 0:
#             for j in range(i+1, i+num_rot+1):
#                 if bitstring[j] == '1':
#                     E_11[t_11][t_11+1] = Q[i][j]
#                     t_11 += 1
#         else:
#             for j in range(i+num_rot, i+num_rot+2):
#                 if bitstring[j] == '1':
#                     E_11[t_11][t_11+1] = Q[i][j]
#                     t_11 += 1

# for i in range(N_res):
#     aad_extended = extended_operator(num_qubits, i, aad)
#     ada_extended = extended_operator(num_qubits, i, ada)
#     H_s += E_1[i] * aad_extended + E_0[i] * ada_extended 

# for i in range(N_res):
#     for j in range(i+1, N_res):
#         aad_extended = extended_operator(num_qubits, i, aad)
#         ada_extended = extended_operator(num_qubits, i, ada)
#         H_i += E_11[i][j] * aad_extended @ aad_extended + \
#                  E_10[i][j] * aad_extended @ ada_extended + \
#                  E_01[i][j] * ada_extended @ aad_extended + \
#                  E_00[i][j] * ada_extended @ ada_extended

s = 0        
for i in range(N_res):
    aad_extended = extended_operator(num_qubits, i, aad)
    ada_extended = extended_operator(num_qubits, i, ada)
    H_s += q[s] * aad_extended + q[s+1] * ada_extended 
    s += 2
    if s >= num:
        break

k = 0
for i in range(N_res):
    aad_extended = extended_operator(num_qubits, i, aad)
    ada_extended = extended_operator(num_qubits, i, ada)
    H_i += v[k] * aad_extended @ aad_extended + \
                v[k+1] * aad_extended @ ada_extended + \
                v[k+2] * ada_extended @ aad_extended + \
                v[k+3] * ada_extended @ ada_extended
    k += 4
    if k >= numm:
        break

H_tt = H_i + H_s 
eigenvalue, eigenvector = eigsh(H_tt, k=num_qubits, which='SA')
print('\nThe ground state with the number operator classically is: ', eigenvalue[0])
print('The classical eigenstate is: ', eigenvalue)

ground_state= eig(H_tt)
print('eig result:', ground_state)


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

for i in range(0, num, num_rot):  #each loop is one residue
    for j in range(0, num_qubits, qubit_per_res):
        N_0i = N_0(j, num_qubits)
        N_1i = N_1(j, num_qubits)
        N_0j = N_0(j+1, num_qubits)
        N_1j = N_1(j+1, num_qubits)
        H_self += q[i] * N_0i @ N_0j + q[i+1] * N_0i @ N_1j + q[i+2] * N_1i @ N_0j + q[i+3] * N_1i @ N_1j 


for i in range(0, numm, 2**num_qubits):     #one loop is one pair of residues
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

            H_int += v[i] * N_0i @ N_0ii @ N_0j @ N_0jj + v[i+1] * N_0i @ N_0ii @ N_0j @ N_1jj + v[i+2] * N_0i @ N_0ii @ N_1j @ N_0jj + v[i+3] * N_0i @ N_0ii @ N_1j @ N_1jj + \
                    + v[i+4] * N_0i @ N_1ii @ N_0j @ N_0jj + v[i+5] * N_0i @ N_1ii @ N_1j @ N_1jj + v[i+6] * N_0i @ N_1ii @ N_1j @ N_0jj + + v[i+7] * N_0i @ N_1ii @ N_1j @ N_1jj + \
                    + v[i+8] * N_1i @ N_0ii @ N_0j @ N_0jj + v[i+9] * N_1i @ N_0ii @ N_0j @ N_1jj + v[i+10] * N_1i @ N_0ii @ N_1j @ N_0jj + v[i+11] * N_1i @ N_0ii @ N_1j @ N_1jj + \
                    + v[i+12] * N_1i @ N_1ii @ N_0j @ N_0jj + v[i+13] * N_1i @ N_1ii @ N_0j @ N_1jj + v[i+14] * N_1i @ N_1ii @ N_1j @ N_0jj + v[i+15] * N_1i @ N_1ii @ N_1j @ N_1jj 
    

H_gen = H_int + H_self


def X_op(i, num):
    op_list = ['I'] * num
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))

mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
p = 10
initial_point = np.ones(2*p)
qaoa1 = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result_gen = qaoa1.compute_minimum_eigenvalue(H_gen)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result_gen.best_measurement)
print('The ground state energy with QAOA is: ', np.real(result_gen.best_measurement['value']))
print(result_gen)

