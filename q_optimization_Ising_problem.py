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
h = np.multiply(0.5,h)       #to convert from QUBO to Ising hamiltonian
num = len(h)

print(f"\nOne body energy values: \n", h)

df = pd.read_csv("two_body_terms.csv")
value = df['E_ij'].values
value = np.multiply(0.5, value)
J = np.zeros((num,num))
n = 0

for i in range(0, num-2):
    if i%2 == 0:
        J[i][i+2]=0.5*value[n]
        J[i][i+3]=0.5*value[n+1]
        n += 2
    elif i%2 != 0:
        J[i][i+1]=0.5*value[n]
        J[i][i+2]=0.5*value[n+1]
        n += 2

print(f"\nTwo body energy values: \n", J)

for i in range(0, num):
    J[i][i] = h[i]

print(f"\nMatrix of all pairwise interaction energies: \n", J)

# add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue
# implementation of the Hammings constraint
def add_penalty_term(J, penalty_constant, residue_pairs):
    for i, j in residue_pairs:
        J[i][j] += penalty_constant
        
    return J


P =3000
residue_pairs = [(0,1), (2,3), (4,5), (6,7)]

J = add_penalty_term(J, P, residue_pairs)

## Classical optimisation:
# function to construct the ising hamiltonain from the one-body and two-body energies h and J
def ising_hamiltonian(config, one_body_energies, two_body_energies):
    hamiltonian = 0

    for i in range(1, num):
        hamiltonian += one_body_energies[i] * config[i]

    for i in range(num):
        for j in range(i+1, num):
            hamiltonian +=two_body_energies[i][j] * config[i]*config[j]
        
    return hamiltonian

# define a random initial configuration
initial_config = np.random.choice([-1, 1], size=num)

# energy function to be minimised
def energy_function(config):
    return ising_hamiltonian(config, h, J)   

## First we will diagonalise classically and then compare the result with the qaoa
# Define the optimization function to minimize the Ising Hamiltonian
from scipy.optimize import minimize
result = minimize(energy_function, initial_config, method='COBYLA')

# Extract the ground state configuration and energy
ground_state_config = result.x
ground_state_energy = result.fun

print("\n\nClassical optimisation results. \n")
print("Ground state energy: ", ground_state_energy)
print("Ground state wavefunction: ", ground_state_config)



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
    pauli_str[i] = 'Z'
    pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))

hamiltonian_terms = []

for i in range(0, num_qubits):
    for j in range(0, num_qubits):
        if J[i][j] != 0:
            pauli = generate_pauli_zij(num_qubits, i, j)
            op = SparsePauliOp(pauli, coeffs=[J[i][j]])
            hamiltonian_terms.append(op) 

hamiltonian = sum(hamiltonian_terms, SparsePauliOp(Pauli('I'*num_qubits)))

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)

print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(hamiltonian))


#the mixer in QAOA should be a quantum operator representing transitions between configurations
mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))


p = 10  # Number of QAOA layers
initial_point = np.ones(2 * p)
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result = qaoa.compute_minimum_eigenvalue(hamiltonian)
print("\n\nThe result of the quantum optimisation using QAOA is: \n", result)
