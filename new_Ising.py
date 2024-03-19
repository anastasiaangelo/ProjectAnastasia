# New hybrid model for encoding 2 rotamers per qubit, so N_rot/2 qubits in total, to reduce circuit depth
# for the case of 1 qubit per residue, so 2 rotamers per residue, check
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

num_rot = 2

df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("energy_files/two_body_terms.csv")
v = df['E_ij'].values
numm = len(v)

num_qubits = N_res

## First Classically
from scipy.sparse.linalg import eigsh

H_s = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
H_i = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
identity = np.eye(2)

a = 0.5*(X + 1j*Y)
a_dagger = 0.5 *(X - 1j*Y) 

aad = a@a_dagger
ada = a_dagger@a

def extended_operator(n, qubit, op):
    ops = [identity if i != qubit else op for i in range(n)]
    extended_op = ops[0]
    for op in ops[1:]:
        extended_op = np.kron(extended_op, op)
    return extended_op

s = 0
for i in range(N_res):
    aad_extended = extended_operator(num_qubits, i, aad)
    ada_extended = extended_operator(num_qubits, i, ada)
    H_s += q[s] * aad_extended + q[s+1] * ada_extended 
    s += num_rot
    if s >= num:
        break

k = 0
for i in range(N_res-1):
    aad_extended = extended_operator(num_qubits, i, aad)
    ada_extended = extended_operator(num_qubits, i, ada)
    aad_extended1 = extended_operator(num_qubits, i+1, aad)
    ada_extended1 = extended_operator(num_qubits, i+1, ada)
    H_i += v[k] * aad_extended @ aad_extended1 + \
                v[k+1] * aad_extended @ ada_extended1 + \
                v[k+2] * ada_extended @ aad_extended1 + \
                v[k+3] * ada_extended @ ada_extended1
    k += num_rot**2
    if k >= numm:
        break

H_tt = H_i + H_s 
eigenvalue, eigenvector = eigsh(H_tt, k=num_qubits, which='SA')
print('\n\nThe ground state with the number operator classically is: ', eigenvalue[0])
# print('The classical eigenstate is: ', eigenvalue)

# ground_state = eig(H_tt)
# print('eig result:', ground_state)


## Mapping to qubits
#  Find minimum value using optimisation technique of QAOA
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

num_qubits = int(num/2)

H_self = SparsePauliOp(Pauli('I'* num_qubits), coeffs=[0])
H_int = SparsePauliOp(Pauli('I'* num_qubits), coeffs=[0]) 

def X_op(i, num_qubits):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = ['I'] * num_qubits
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))

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

l = 0
for i in range(N_res):
    N_0i = N_0(i, num_qubits)
    N_1i = N_1(i, num_qubits)
    H_self += q[l] * N_0i + q[l+1] * N_1i 
    l += 2
    if l >= num:
        break

j = 0
for i in range(N_res-1):
    N_0i = N_0(i, num_qubits)
    N_1i = N_1(i, num_qubits)
    N_0j = N_0(i+1, num_qubits)
    N_1j = N_1(i+1, num_qubits)
    H_int += v[j] * N_0i @ N_0j + v[j+1] * N_0i @ N_1j + v[j+2] * N_1i @ N_0j + v[j+3] * N_1i @ N_1j
    j += 4
    if j >= numm:
        break

H_gen = H_int + H_self

mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
p = 1
initial_point = np.ones(2*p)
qaoa1 = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result_gen = qaoa1.compute_minimum_eigenvalue(H_gen)
print("\nThe result of the quantum optimisation using QAOA with the number operators is: \n")
print('best measurement', result_gen.best_measurement)
print('\nThe ground state energy with QAOA is: ', np.real(result_gen.best_measurement['value']), '\n')
print(result_gen)

