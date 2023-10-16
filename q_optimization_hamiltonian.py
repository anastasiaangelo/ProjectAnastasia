# Script to optimise the Hamiltonian, starting with the QUBO problem, transforming to Ising problem, optimising
from qiskit import Aer, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.opflow.expectations import PauliExpectation
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import VQE


# Define the Ising Hamiltonian coefficients
h = [0.5, -1.0]             #these will be the Jii values, so the one body energies
J = [[0.0, 2.0], [2.0, 0.0]] # this will be imported from the Ising_hamiltonian scripts, Jij two body energies

# Create a Qiskit Quantum Circuit: Initialize a quantum circuit that will be used to represent 
# the Hamiltonian. You will apply the terms of the Hamiltonian as quantum gates.
num_qubits = len(h)
circ = QuantumCircuit(num_qubits)

#Apply Pauli-Z Gates (spin variables)
for i in range(num_qubits):
    circ.rz(2 * h[i], i)


#Apply CNOT Gates: For each term JijZiZj, apply a CNOT gate (controlled-X gate) with qubit i
# as the control and qubit j as the target. You may need to adjust the sign of Jij based on your specific definition of the Ising model.
for i in range(num_qubits):
    for j in range(i + 1, num_qubits):
        circ.cx(i, j)
        circ.rz(2 * J[i][j], j)
        circ.cx(i, j)

#Measure the expectation value by executing the circuit multiple times and taking the average. The expectation value provides an estimate of the energy of the current quantum state.
# Define the quantum simulator
simulator = AerSimulator()

# Convert the circuit to an executable form
compiled_circuit = transpile(circ, simulator)

# # Create a function to compute the expectation value
# param = Parameter('theta')
# expectation = PauliExpectation().convert(compiled_circuit.bind_parameters({param: 0.0}))

# Create a PauliSumOp operator from the circuit
op = PauliSumOp.from_list(circ)

# Create a Quadratic Program with the expectation value
qubo = QuadraticProgram()
qubo.from_ising(expectation, offset=0.0)


#Create a QAOA based on the quadratic program which approximates the ground state of the Ising Hamiltonian
# using here a simulator, this can be replaced with a quantum device if we want
qaoa = MinimumEigenOptimizer(QAOA(quantum_instance=Aer.get_backend('qasm_simulator')))
result = qaoa.solve(qubo)

# # Create a VQE optimizer and minimize the expectation value -- alternative minimization algorithm
# optimizer = VQE(expectation, optimizer=COBYLA())
# result = optimizer.solve(qubo)

#Analyse the result to get the optimal solution for the Ising problem
print('Optimal solution:', result.x)
print('Optimal value:', result.fval)
