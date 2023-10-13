# optimization experiment to compute the solution of a Max-Cut which can be formulated as a quadratic program
#the MinimumEigenOptimizer is employed in combination with the Quantum Approximate Optimization Algorithm (QAOA) as minimum eigensolver routine.

from docplex.mp.model import Model

from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp

from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import SPSA

# Generate a graph of 4 nodes
n = 4
edges = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]  # (node_i, node_j, weight)

# Formulate the problem as a Docplex model
model = Model()

# Create n binary variables
x = model.binary_var_list(n)

# Define the objective function to be maximized
model.maximize(model.sum(w * x[i] * (1 - x[j]) + w * (1 - x[i]) * x[j] for i, j, w in edges))

# Fix node 0 to be 1 to break the symmetry of the max-cut solution
model.add(x[0] == 1)

# Convert the Docplex model into a `QuadraticProgram` object
problem = from_docplex_mp(model)

# Run quantum algorithm QAOA on qasm simulator
seed = 1234
algorithm_globals.random_seed = seed

spsa = SPSA(maxiter=250)
sampler = Sampler()
qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=5)
algorithm = MinimumEigenOptimizer(qaoa)
result = algorithm.solve(problem)
print(result.prettyprint())  # prints solution, x=[1, 0, 1, 0], the cost, fval=4
