import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from copy import deepcopy

num_rot = 2

df1 = pd.read_csv("one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N = int(num/num_rot)

H_self = np.zero((num, num))
H_int = np.zeros((num, num))

a = np.array([[0, 1], [0, 0]])
a_dagger = np.array([[0, 0], [1, 0]])
I = np.eye(2)

from qiskit import Aer, QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

#function to generate operators for n qubits, given n qubits and q target qubit, it returns the appropriate annihilation or creation operator extended to the n-qubit system
def extended_operator(n, qubit, op):
    ops = [I if i != qubit else op for i in range(n)]
    extended_op = ops[0]
    for op in ops[1:]:
        extended_op = np.kron(extended_op, op)
    return extended_op

num_qubits = N
# to apply the annihilation/creation operators to all qubits in the system
for i in range(num_qubits):
    for j in range(i+1, num_qubits):
        operator = extended_operator(num_qubits, i, a)
        