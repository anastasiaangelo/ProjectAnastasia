# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
import numpy as np
import csv

## here we will use the classical code to calculate and import the matrix J
# use sparse representation for memory reasons
with open('hamiltonian_terms.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    data = []
    for row in csv_reader:
        complex_row = []
        for entry in row:
            try:
                complex_row.append(complex(entry))
            except ValueError:
              print(f"Error converting string {entry} to complex. Please check the CSV.")
              complex_row.append(0j)  # add a default value; or you can skip this row/entry depending on your needs
        data.append(complex_row)
    J = np.array(data)  


# or build the Pauli representation from the problem may be more efficient rather than converting it

from qiskit import Aer, QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp, MatrixOp
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Operator
from qiskit.algorithms.optimizers import COBYLA

# Convert MatrixOp to PauliOp
pauli_op = MatrixOp(J).to_pauli_op()
# Extract Pauli terms and coefficients from PauliOp
pauli_terms = [(str(term.primitive), term.coeff) for term in pauli_op]
# Convert to PauliOp
hamiltonian = PauliSumOp.from_list(pauli_terms)

print(hamiltonian)


## find minimum value using optimisation technique of QAOA
nqubits = 4

qaoa = QAOA(1,optimizer=COBYLA(), reps=1, mixer=hamiltonian, initial_point=[1.0,1.0])
result = qaoa.compute_minimum_eigenvalue(hamiltonian)
print(result)
