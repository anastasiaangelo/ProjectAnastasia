import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from copy import deepcopy

num_rot = 2

df1 = pd.read_csv("one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("two_body_terms.csv")
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
num_qubits = num

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

eigenvalues, eigenvectors = eigsh(H_tot, k=num, which='SA')

## Quantum optimisation
#  Find minimum value using optimisation technique of QAOA
from qiskit import Aer, QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def X_op(i, num_qubits):
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

p = 10  # Number of QAOA layers
initial_point = np.ones(2 * p)
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result.best_measurement)

k = 0
for i in range(num_qubits):
    k += 0.5 * q[i]

for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            k += 0.5 * 0.25 * Q[i][j]

print('The ground state energy classically is: ', eigenvalues[0] + N_res*P + k)
print('The ground state energy with QAOA is: ', np.real(result.best_measurement['value']) + N_res*P + k)

bitstring = result.best_measurement['bitstring']
E_0 = np.zeros(N_res)
E_1 = np.zeros(N_res)

t_0 = 0
t_1 = 0

for i in range(num):
    if bitstring[i] == '0' and t_0 < N_res:
        E_0[t_0] = q[i]
        t_0 += 1
    elif bitstring[i] == '1' and t_1 < N_res:
        E_1[t_1] = q[i]
        t_1 += 1

print('zero bitstirng energy values: \n', E_0)
print('one bitstirng energy values: \n', E_1)
 
H_self = np.zeros((num, num))
H_int = np.zeros((num, num))

a = np.array([[0, 1], [0, 0]])
a_dagger = np.array([[0, 0], [1, 0]])
I = np.eye(2)

#function to generate operators for n qubits, given n qubits and q target qubit, it returns the appropriate annihilation or creation operator extended to the n-qubit system
def extended_operator(n, qubit, op):
    ops = [I if i != qubit else op for i in range(n)]
    extended_op = ops[0]
    for op in ops[1:]:
        extended_op = np.kron(extended_op, op)
    return extended_op

num_qubits = N_res
# to apply the annihilation/creation operators to all qubits in the system
for i in range(num_qubits):
    for j in range(i+1, num_qubits):
        operator = extended_operator(num_qubits, i, a)
        

for i in range(num_qubits-1):
    H_self = E_0[i] * a * a_dagger + E_1[i] * a_dagger * a
