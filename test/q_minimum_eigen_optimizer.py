#Converting a QUBO problem to an Ising problem -- first possible approach
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
import matplotlib.pyplot as plt

## Converting a QUBO to an operator
# create a QUBO
qubo = QuadraticProgram()
qubo.binary_var("x")
qubo.binary_var("y")
qubo.binary_var("z")
qubo.minimize(linear=[1, -2, 3], quadratic={("x", "y"): 1, ("x", "z"): -1, ("y", "z"): 2})
print(qubo.prettyprint())

#Next we translate this QUBO into an Ising operator. This results in an Operator and also a constant offset to 
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
# initializing the MinimumEigensolver we want to use.
algorithm_globals.random_seed = 10598       #sets the random seed for the random number generator used by the quantum algorithms in Qiskit. Setting a random seed ensures that the algorithm's behavior is reproducible
qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])    #creates a Quantum Approximate Optimization Algorithm (QAOA) instance
exact_mes = NumPyMinimumEigensolver()       #creates an instance of the NumPyMinimumEigensolver, a classical algorithm for solving for the minimum eigenvalue and eigenvector of a matrix

# we use the MinimumEigensolver to create MinimumEigenOptimizer
qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA, creating an instance of the class
exact = MinimumEigenOptimizer(exact_mes)  # using the exact classical numpy minimum eigen solver, creating an instance of the class

#We first use the MinimumEigenOptimizer based on the classical exact NumPyMinimumEigensolver to get the optimal benchmark solution for this small example.
exact_result = exact.solve(qubo)
print(exact_result.prettyprint())

# Next we apply the MinimumEigenOptimizer based on QAOA to the same problem.
qaoa_result = qaoa.solve(qubo)
print(qaoa_result.prettyprint())

## Analysis of samples
#Multiple samples corresponding to the same input are consolidated into a single SolutionSample
print("variable order:", [var.name for var in qaoa_result.variables])
for s in qaoa_result.samples:
    print(s)

#you may want to filter samples according to their status or probabilities
def get_filtered_samples(
    samples: List[SolutionSample],
    threshold: float = 0,
    allowed_status: Tuple[OptimizationResultStatus] = (OptimizationResultStatus.SUCCESS,),
):
    res = []
    for s in samples:
        if s.status in allowed_status and s.probability > threshold:
            res.append(s)

    return res

filtered_samples = get_filtered_samples(
    qaoa_result.samples, threshold=0.005, allowed_status=(OptimizationResultStatus.SUCCESS,)
)
for s in filtered_samples:
    print(s)


# to obtain a better insight of the samples -> statistics
fvals = [s.fval for s in qaoa_result.samples]
probabilities = [s.probability for s in qaoa_result.samples]
np.mean(fvals)
np.std(fvals)

#visualisation
samples_for_plot = {
    " ".join(f"{qaoa_result.variables[i].name}={int(v)}" for i, v in enumerate(s.x)): s.probability
    for s in filtered_samples
}
samples_for_plot

plot_histogram(samples_for_plot)
plt.show()

## RecursiveMinimumEingenOptimizer
#eakes a MinimumEigenOptimizer as input and applies the recursive optimization scheme to reduce the size of the problem one variable at a time.
#construct the RecursiveMinimumEigenOptimizer such that it reduces the problem size from 3 variables to 1 variable and then uses the exact solver for the last variable
rqaoa = RecursiveMinimumEigenOptimizer(qaoa, min_num_vars=1, min_num_vars_optimizer=exact)
rqaoa_result = rqaoa.solve(qubo)
print(rqaoa_result.prettyprint())

filtered_samples = get_filtered_samples(
    rqaoa_result.samples, threshold=0.005, allowed_status=(OptimizationResultStatus.SUCCESS,)
)

samples_for_plot = {
    " ".join(f"{rqaoa_result.variables[i].name}={int(v)}" for i, v in enumerate(s.x)): s.probability
    for s in filtered_samples
}
samples_for_plot

plot_histogram(samples_for_plot)
plt.show()
