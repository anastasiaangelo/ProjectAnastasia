# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
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
def ising_hamiltonian(one_body_energies, two_body_energies):        # we can then add config when we want to introduce the spin variable
    hamiltonian = 0

    for i in range(1, num):
        hamiltonian += one_body_energies[i]     # * config[i]

    for i in range(num):
        for j in range(i+1, num):
            hamiltonian +=two_body_energies[i][j]           #*config[i]*config[j]
        
    return hamiltonian


#define a random initial configuration
initial_config = np.array([1, -1, -1, 1, 1, 1]) 


# or build the Pauli representation from the problem may be more efficient rather than converting it
# too complex though for now

from qiskit import Aer, QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp, MatrixOp
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp

hamiltonian = ising_hamiltonian(h, J)       #initial_config

## find minimum value using optimisation technique of QAOA

qaoa = QAOA(1, optimizer=COBYLA(), reps=1, mixer=hamiltonian, initial_point=[1.0,1.0])
result = qaoa.compute_minimum_eigenvalue(hamiltonian)
print("\n\nthe result is", result)
