# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# or build the Pauli representation from the problem may be more efficient rather than converting it
# too complex though for now 
import numpy as np
import pandas as pd
import csv

## configure the hamiltonian from the values calculated classically with pyrosetta
df1 = pd.read_csv("one_body_terms.csv")
h = df1['E_ii'].values
num = len(h)

print(h)

df = pd.read_csv("two_body_terms.csv")
value = df['E_ij'].values
J = np.zeros((num,num))
n = 0
print(value)

for i in range(0, num-2):
    if i%2 == 0:
        J[i][i+2]=value[n]
        J[i][i+3]=value[n+1]
        n += 2
    elif i%2 != 0:
        J[i][i+1]=value[n]
        J[i][i+2]=value[n+1]
        n += 2

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
from qiskit.opflow import PauliSumOp, MatrixOp, I, X, Z
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp

def Z_op(qubit, num):
    """Return a Z Pauli operator on the specified qubit in a num-qubit system."""
    op_list = [I for _ in range(num)]
    op_list[qubit] = Z
    op = op_list[0]
    for current_op in op_list[1:]:
        op = op @ current_op
    return op

def X_op(qubit, num):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = [I for _ in range(num)]
    op_list[qubit] = X
    op = op_list[0]
    for current_op in op_list[1:]:
        op = op @ current_op
    return op

def ising_to_pauli_op(h, J):
    num = len(h)
    op = sum(h[i] * Z_op(i, num) for i in range(num))
    for i in range(num):
        for j in range(i+1, num):
            op += J[i][j] * Z_op(i, num) @ Z_op(j, num)
    return op

hamiltonian_op = ising_to_pauli_op(h, J)

#the mixer in QAOA should be a quantum operator representing transitions between configurations
mixer_op = sum(X_op(i,num) for i in range(num))

hamiltonian = ising_hamiltonian(initial_config, h, J)  

qaoa = QAOA(1, optimizer=COBYLA(), reps=1, mixer=mixer_op, initial_point=[1.0,1.0])
result = qaoa.compute_minimum_eigenvalue(hamiltonian_op)
print("\n\nthe result is", result)