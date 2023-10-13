# Script to optimise the Hamiltonian, starting with the QUBO problem, transforming to Ising problem, optimising
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

## Converting a QUBO to an operator
# create a QUBO with two qubits for example
qubo = QuadraticProgram()
qubo.binary_var("x0")
qubo.binary_var("x1")
qubo.minimize(linear=[1, 1], quadratic={('x0', 'x1'): 2})

#Create a QAOA based on the quadratic program which approximates the ground state of the Ising Hamiltonian
# using here a simulator, this can be replaced with a quantum device if we want
qaoa = MinimumEigenOptimizer(QAOA(quantum_instance=Aer.get_backend('qasm_simulator')))
result = qaoa.solve(qubo)

#Analyse the result to get the optimal solution for the Ising problem
print('Optimal solution:', result.x)
print('Optimal value:', result.fval)
