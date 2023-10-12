# Quadratic programs
import numpy as np 
import scipy

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

simulator = AerSimulator()

circuit = QuantumCircuit(2,2)
circuit.h(0)
circuit.cx(0,1)
circuit.measure([0,1], [0,1])

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shot=1000)
result = job.result()

counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:", counts)

circuit.draw("mpl")
