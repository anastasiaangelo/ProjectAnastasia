# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
import numpy as np

## here we will use the classical code to calculate and import the matrix J
# use sparse representation for memory reasons
J = np.array([[1,2], [3,4]])
# or build the Pauli representation from the problem may be more efficient rather than converting it

from qiskit import Aer, QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp, MatrixOp, Z, I
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

H = -1.0 * (Z ^ Z ^ I ^ I) - 1.0 * (I ^ Z ^ Z ^ I) - 1.0 * (I ^ I ^ Z ^ Z)


## find minimum value using optimisation technique of QAOA
nqubits = 4

qaoa = QAOA(1,optimizer=COBYLA(), reps=1, mixer=H, initial_point=[1.0,1.0])
result = qaoa.compute_minimum_eigenvalue(H)
print(result)
