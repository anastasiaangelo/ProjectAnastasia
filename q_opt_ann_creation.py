import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
import cmath

num_rot = 2

df1 = pd.read_csv("one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("two_body_terms.csv")
v = df['E_ij'].values
numm = len(v)
Q = np.zeros((num,num))
n = 0

for i in range(num-2):
    if i%2 == 0:
        Q[i][i+2] = deepcopy(v[n])
        Q[i+2][i] = deepcopy(v[n])
        Q[i][i+3] = deepcopy(v[n+1])
        Q[i+3][i] = deepcopy(v[n+1])
        n += 2
    elif i%2 != 0:
        Q[i][i+1] = deepcopy(v[n])
        Q[i+1][i] = deepcopy(v[n])
        Q[i][i+2] = deepcopy(v[n+1])
        Q[i+2][i] = deepcopy(v[n+1])
        n += 2

print("q: \n", q)
print("Q: \n", Q)
H = np.zeros((num,num))

for i in range(num):
    for j in range(num):
        if i != j:
            H[i][j] = np.multiply(0.25, Q[i][j])

for i in range(num):
    H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

# add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
def add_penalty_term(M, penalty_constant, residue_pairs):
    for i, j in residue_pairs:
        M[i][j] += penalty_constant
    return M

def generate_pairs(N):
    pairs = [(i, i+1) for i in range(0, 2*N, 2)]
    return pairs

P = 6
pairs = generate_pairs(N_res)

M = deepcopy(H)
M = add_penalty_term(M, P, pairs)


## Classical optimisation:
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig

Z_matrix = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

def construct_operator(qubit_indices, num):
    operator = np.eye(1)
    for qubit in range(num):
        if qubit in qubit_indices:
            operator = np.kron(operator, Z_matrix)
        else:
            operator = np.kron(operator, identity)
    return operator

C = np.zeros((2**num, 2**num))

for i in range(num):
    operator = construct_operator([i], num)
    C += H[i][i] * operator

for i in range(num):
    for j in range(i+1, num):
        operator = construct_operator([i, j], num)
        C += H[i][j] * operator

def create_hamiltonian(pairs, P, num):
    H_pen = np.zeros((2**num, 2**num))
    def tensor_term(term_indices):
        term = [Z_matrix if i in term_indices else identity for i in range(num)]
        result = term[0]
        for t in term[1:]:
            result = np.kron(result, t)
        return result
    
    for pair in pairs:
        term = tensor_term(pair)
        H_pen += P * term

    return H_pen

H_penalty = create_hamiltonian(pairs, P, num)
H_tot = C + H_penalty

eigenvalues, eigenvectors = eigsh(H_tot, k=num, which='SA')

## Quantum optimisation
#  Find minimum value using optimisation technique of QAOA
from qiskit import Aer, QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def X_op(i, num):
    op_list = ['I'] * num
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

q_hamiltonian = SparsePauliOp(Pauli('I'*num), coeffs=[0])

for i in range(num):
    for j in range(i+1, num):
        if M[i][j] != 0:
            pauli = generate_pauli_zij(num, i, j)
            op = SparsePauliOp(pauli, coeffs=[M[i][j]])
            q_hamiltonian += op

for i in range(num):
    pauli = generate_pauli_zij(num, i, i)
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

mixer_op = sum(X_op(i,num) for i in range(num))

p = 10  # Number of QAOA layers
initial_point = np.ones(2 * p)
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result.best_measurement)

k = 0
for i in range(num):
    k += 0.5 * q[i]

for i in range(num):
    for j in range(num):
        if i != j:
            k += 0.5 * 0.25 * Q[i][j]

print('\nThe ground state energy classically is: ', eigenvalues[0] + N_res*P + k)
print('The ground state energy with QAOA is: ', np.real(result.best_measurement['value']) + N_res*P + k)


## Creation and Annihilation operators Hamiltonian
num_qubits = N_res

## First Classically

H_s = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
H_i = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

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


bitstring = result.best_measurement['bitstring']

for i in range(num):
    Q[i][i] = deepcopy(q[i])

E_0 = np.zeros(num_qubits)
E_1 = np.zeros(num_qubits)

t_0 = 0
t_1 = 0

for i in range(num):
    if bitstring[i] == '0' and t_0 < num_qubits:
        E_0[t_0] = Q[i][i]
        t_0 += 1
    elif bitstring[i] == '1' and t_1 < num_qubits:
        E_1[t_1] = Q[i][i]
        t_1 += 1

E_00 = np.zeros((num_qubits, num_qubits))
E_01 = np.zeros((num_qubits, num_qubits))
E_10 = np.zeros((num_qubits, num_qubits))
E_11 = np.zeros((num_qubits, num_qubits))

t_00 = 0
t_01 = 0
t_10 = 0
t_11 = 0

for i in range(num-num_rot):

    if bitstring[i] == '0' and t_00 < N_res:
        if i % 2 != 0:
            for j in range(i+1, i+num_rot+1):
                if bitstring[j] == '0':
                    E_00[t_00][t_00+1] = Q[i][j]
                    t_00 += 1
        else:
            for j in range(i+num_rot, i+num_rot+2):
                if bitstring[j] == '0':
                    E_00[t_00][t_00+1] = Q[i][j]
                    t_00 += 1

    if bitstring[i] == '0' and t_01 < N_res:
        if i % 2 != 0:
            for j in range(i+1, i+num_rot+1):
                if bitstring[j] == '1':
                    E_01[t_01][t_01+1] = Q[i][j]
                    t_01 += 1
        else:
            for j in range(i+num_rot, i+num_rot+2):
                if bitstring[j] == '1':
                    E_01[t_01][t_01+1] = Q[i][j]
                    t_01 += 1

    if bitstring[i] == '1' and t_10 < N_res:
        if i % 2 != 0:
            for j in range(i+1, i+num_rot+1):
                if bitstring[j] == '0':
                    E_10[t_10][t_10+1] = Q[i][j]
                    t_10 += 1
        else:
            for j in range(i+num_rot, i+num_rot+2):
                if bitstring[j] == '0':
                    E_10[t_10][t_10+1] = Q[i][j]
                    t_10 += 1

    if bitstring[i] == '1' and t_11 < N_res:
        if i % 2 != 0:
            for j in range(i+1, i+num_rot+1):
                if bitstring[j] == '1':
                    E_11[t_11][t_11+1] = Q[i][j]
                    t_11 += 1
        else:
            for j in range(i+num_rot, i+num_rot+2):
                if bitstring[j] == '1':
                    E_11[t_11][t_11+1] = Q[i][j]
                    t_11 += 1

for i in range(N_res):
    aad_extended = extended_operator(num_qubits, i, aad)
    ada_extended = extended_operator(num_qubits, i, ada)
    H_s += E_1[i] * aad_extended + E_0[i] * ada_extended 

for i in range(N_res):
    for j in range(i+1, N_res):
        aad_extended = extended_operator(num_qubits, i, aad)
        ada_extended = extended_operator(num_qubits, i, ada)
        H_i += E_11[i][j] * aad_extended @ aad_extended + \
                 E_10[i][j] * aad_extended @ ada_extended + \
                 E_01[i][j] * ada_extended @ aad_extended + \
                 E_00[i][j] * ada_extended @ ada_extended

# s = 0        
# for i in range(N_res):
#     aad_extended = extended_operator(num_qubits, i, aad)
#     ada_extended = extended_operator(num_qubits, i, ada)
#     H_s += q[s] * aad_extended + q[s+1] * ada_extended 
#     s += 2
#     if s >= num:
#         break

# k = 0
# for i in range(N_res):
#     aad_extended = extended_operator(num_qubits, i, aad)
#     ada_extended = extended_operator(num_qubits, i, ada)
#     H_i += v[k] * aad_extended @ aad_extended + \
#                 v[k+1] * aad_extended @ ada_extended + \
#                 v[k+2] * ada_extended @ aad_extended + \
#                 v[k+3] * ada_extended @ ada_extended
#     k += 4
#     if k >= numm:
#         break

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

l=0
for i in range(num_qubits):
    N_0i = N_0(i, num_qubits)
    N_1i = N_1(i, num_qubits)
    H_self += q[l] * N_0i + q[l+1] * N_1i 
    l += 2
    if l >= num:
        break  

j = 0
for i in range(num_qubits):
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
p = 10
initial_point = np.ones(2*p)
qaoa1 = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result_gen = qaoa1.compute_minimum_eigenvalue(H_gen)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result_gen.best_measurement)
print('The ground state energy with QAOA is: ', np.real(result_gen.best_measurement['value']))
print(result_gen)

