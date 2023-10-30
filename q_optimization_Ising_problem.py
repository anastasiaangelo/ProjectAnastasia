# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# or build the Pauli representation from the problem may be more efficient rather than converting it
# too complex though for now 
import numpy as np
import pandas as pd
import csv

## configure the hamiltonian from the values calculated classically with pyrosetta
df1 = pd.read_csv("one_body_terms.csv")
h = df1['E_ii'].values
h = np.multiply(0.5,h)       #to convert from QUBO to Ising hamiltonian
num = len(h)

print(h)

df = pd.read_csv("two_body_terms.csv")
value = df['E_ij'].values
value = np.multiply(0.5, value)
J = np.zeros((num,num))
n = 0
print(value)

for i in range(0, num-2):
    if i%2 == 0:
        J[i][i+2]=0.5*value[n]
        J[i][i+3]=0.5*value[n+1]
        n += 2
    elif i%2 != 0:
        J[i][i+1]=0.5*value[n]
        J[i][i+2]=0.5*value[n+1]
        n += 2

print(J)

for i in range(0, num):
    J[i][i] = h[i]

print(J)

## function to construct the ising hamiltonain from the one-body and two-body energies h and J
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

print("Ground state energy: ", ground_state_energy)
print("Ground state wavefunction: ", ground_state_config)



## Find minimum value using optimisation technique of QAOA
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp, MatrixOp, I, X, Z, PauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp

num_qubits = 4

def Z_op(qubit, num_qubits):
    """Return a Z Pauli operator on the specified qubit in a num-qubit system."""
    op_list = [I for _ in range(num_qubits)]
    op_list[qubit] = Z
    op = op_list[0]
    for current_op in op_list[1:]:
        op = op @ current_op
    return op

def X_op(qubit, num_qubits):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = [I for _ in range(num_qubits)]
    op_list[qubit] = X
    op = op_list[0]
    for current_op in op_list[1:]:
        op = op @ current_op
    return op

def ising_to_pauli_op(h, J):
    num = len(h)
    op = sum(h[i] * Z_op(i, num_qubits) for i in range(num_qubits))
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            op += J[i][j] * Z_op(i, num_qubits) @ Z_op(j, num_qubits)
    return op
 

def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
   
    pauli_str = ['I']*n
    pauli_str[i] = 'Z'
    pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))

hamiltonian_terms = []

for i in range(0, num):
    for j in range(0, num):
        if J[i][j] != 0:
            op = PauliOp(generate_pauli_zij(2**num_qubits, i, j), coeff=J[i][j])
            hamiltonian_terms.append(op) 

hamiltonian = sum(hamiltonian_terms)

print(hamiltonian)


#the mixer in QAOA should be a quantum operator representing transitions between configurations
mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))

qaoa = QAOA(1, optimizer=COBYLA(), reps=1, mixer=mixer_op, initial_point=[1.0,1.0])
result = qaoa.compute_minimum_eigenvalue(hamiltonian)
print("\n\nthe result is", result)