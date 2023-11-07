# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# or build the Pauli representation from the problem may be more efficient rather than converting it
# too complex though for now 
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

## configure the hamiltonian from the values calculated classically with pyrosetta
df1 = pd.read_csv("one_body_terms.csv")
h = df1['E_ii'].values
h = np.multiply(0.5, h)       #to convert from QUBO to Ising hamiltonian
num = len(h)

print(f"\nOne body energy values: \n", h)

df = pd.read_csv("two_body_terms.csv")
value = df['E_ij'].values
J = np.zeros((num,num))
n = 0

for i in range(0, num-2):
    if i%2 == 0:
        J[i][i+2] = 0.25 * value[n]
        J[i][i+3] = 0.25 * value[n+1]
        n += 2
    elif i%2 != 0:
        J[i][i+1] = 0.25 * value[n]
        J[i][i+2] = 0.25 * value[n+1]
        n += 2

print(f"\nTwo body energy values: \n", J)

H = J
for i in range(0, num):
    H[i][i] = h[i]

print(f"\nMatrix of all pairwise interaction energies: \n", H)


# add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
def add_penalty_term(M, penalty_constant, residue_pairs):
    for i, j in residue_pairs:
        M[i][j] += penalty_constant
        
    return M

P = 10
residue_pairs = [(0,1), (2,3), (4,5), (6,7)]     #, (8,9), (10,11), (12,13)]

M = add_penalty_term(H, P, residue_pairs)

# constructing Ising as outlined in Ben's pdf
def ising_hamiltonian(config, one_body_energies, two_body_energies):
    hamiltonian = 0
    num = len(one_body_energies)

    for i in range(num):
        for j in range(i+1, num):
            hamiltonian += two_body_energies[i][j] * config[i] * config[j] 
    
    for i in range(num):
        for j in range(num):
            hamiltonian -= two_body_energies[i][j] * config[i]

    for i in range(num):
        for j in range(num):
            hamiltonian -= two_body_energies[i][j] * config[j]

    for i in range(num):
        hamiltonian -= one_body_energies[i] * config[i]

    # for i in range(num):
    #     for j in range(num):
    #          hamiltonian += two_body_energies[i][j]
    
    # for i in range(num):
    #     hamiltonian += one_body_energies[i]
        
    return hamiltonian



# define a random initial configuration
initial_config = np.random.choice([-1, 1], size=num)

## Classical optimisation:
from scipy.sparse.linalg import eigsh

eigenvalues, eigenvectors = eigsh(M, k=num, which='SA')

print("\n\nClassical optimisation results. \n")
print("Ground energy eigsh: ", eigenvalues[0])
print("ground state wavefuncion eigsh: ", eigenvectors[:,0])


## Quantum optimisation
#  Find minimum value using optimisation technique of QAOA
from qiskit import Aer, QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

num_qubits = num

def X_op(i, num_qubits):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = ['I'] * num_qubits
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))


def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
   
    pauli_str = ['I']*n

    if i ==j:
        pauli_str[i] = 'Z'
    else:
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))

hamiltonian_terms = []

for i in range(num_qubits):
    for j in range(num_qubits):
        if M[i][j] != 0:
            pauli = generate_pauli_zij(num_qubits, i, j)
            op = SparsePauliOp(pauli, coeffs=[M[i][j]])
            hamiltonian_terms.append(op) 

zero_op = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])
q_hamiltonian = sum(hamiltonian_terms, zero_op)

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)

print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))


#the mixer in QAOA should be a quantum operator representing transitions between configurations
mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))


p = 10  # Number of QAOA layers
initial_point = np.ones(2 * p)
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('eigenvalue: ', result.eigenvalue)
print('best measurement', result.best_measurement)
print(result)

k = 0
for i in range(num):
    k += h[i]
for i in range(num):
    k += 0.25 * value[i]

print(k)

print('Ground state energy quantum: ', result.eigenvalue + k)