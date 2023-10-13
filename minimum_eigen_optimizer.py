#Converting a QUBO problem to an Ising problem
from qiskit.utils import algorithm_globals
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np

## Converting a QUBO to an operator
# create a QUBO
qubo = QuadraticProgram()
qubo.binary_var("x")
qubo.binary_var("y")
qubo.binary_var("z")
qubo.minimize(linear=[1, -2, 3], quadratic={("x", "y"): 1, ("x", "z"): -1, ("y", "z"): 2})
print(qubo.prettyprint())

#Next we translate this QUBO into an Ising operator. This results not only in an Operator but also in a constant offset to 
# be taken into account to shift the resulting value.
op, offset = qubo.to_ising()
print("offset: {}".format(offset))
print("operator:")
print(op)

#Sometimes a QuadraticProgram might also directly be given in the form of an Operator. 
# For such cases, Qiskit also provides a translator from an Operator back to a QuadraticProgram
qp = QuadraticProgram()
qp.from_ising(op, offset, linear=True)
print(qp.prettyprint())

#This translator allows, for instance, one to translate an Operator to a QuadraticProgram and then solve the problem with other 
# algorithms that are not based on the Ising Hamiltonian representation, such as the GroverOptimizer.


## Solving a QUBO with the MinimumEigenOptimizer
#  initializing the MinimumEigensolver we want to use.
algorithm_globals.random_seed = 10598
qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
exact_mes = NumPyMinimumEigensolver()

# we use the MinimumEigensolver to create MinimumEigenOptimizer
qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
exact = MinimumEigenOptimizer(exact_mes)  # using the exact classical numpy minimum eigen solver

#We first use the MinimumEigenOptimizer based on the classical exact NumPyMinimumEigensolver to get the optimal benchmark solution for this small example.
exact_result = exact.solve(qubo)
print(exact_result.prettyprint())

# Next we apply the MinimumEigenOptimizer based on QAOA to the same problem.
qaoa_result = qaoa.solve(qubo)
print(qaoa_result.prettyprint())

