# First qiskit script
import numpy as np 
import scipy, setuptools, networkx, docplex, qiskit

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

# Design a quantum circuit, compile the circuit and run it on specified quantum services, compute summart statistics
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

### Quadratic programs: how to build optimization problems using Qiskit’s optimization module

# make a Docplex model
from docplex.mp.model import Model

mdl = Model("docplex model")
x = mdl.binary_var("x")
y = mdl.integer_var(lb=-1, ub=5, name="y")
mdl.minimize(x + 2 * y)         #quadratic problem to minimize
mdl.add_constraint(x - y == 3)  #constraints
mdl.add_constraint((x + y) * (x - y) <= 1)
print(mdl.export_as_lp_string())

# load from a Docplex model
mod = from_docplex_mp(mdl)
print(type(mod))
print()
print(mod.prettyprint())    # to generate a comprehensive string representation


## Directly constructing a QuadraticProgram

# make an empty problem
mod = QuadraticProgram("my problem")
print(mod.prettyprint())

# Add variables
mod.binary_var(name="x")
mod.integer_var(name="y", lowerbound=-1, upperbound=5)
mod.continuous_var(name="z", lowerbound=-1, upperbound=5)
print(mod.prettyprint())
# for quadratic programs, there are 3 pieces that have to be specified: a constant (offset), a linear term (cTx), and a quadratic term (xTQx)

# Add objective function using dictionaries
mod.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2, ("z", "z"): -1})
print(mod.prettyprint())

#Alternative: Add objective function using lists/arrays
mod.minimize(constant=3, linear=[1, 0, 0], quadratic=[[0, 1, 0], [1, 0, 0], [0, 0, -1]])
print(mod.prettyprint())

#Access the constant, linear and quadratic terms
print("constant:\t\t\t", mod.objective.constant)
print("linear dict:\t\t\t", mod.objective.linear.to_dict())
print("linear array:\t\t\t", mod.objective.linear.to_array())
print("linear array as sparse matrix:\n", mod.objective.linear.coefficients, "\n")
print("quadratic dict w/ index:\t", mod.objective.quadratic.to_dict())
print("quadratic dict w/ name:\t\t", mod.objective.quadratic.to_dict(use_name=True))
print(
    "symmetric quadratic dict w/ name:\t",
    mod.objective.quadratic.to_dict(use_name=True, symmetric=True),
)
print("quadratic matrix:\n", mod.objective.quadratic.to_array(), "\n")
print("symmetric quadratic matrix:\n", mod.objective.quadratic.to_array(symmetric=True), "\n")
print("quadratic matrix as sparse matrix:\n", mod.objective.quadratic.coefficients)

#Adding/removing linear and quadratic constraints

# Add linear constraints
mod.linear_constraint(linear={"x": 1, "y": 2}, sense="==", rhs=3, name="lin_eq")
mod.linear_constraint(linear={"x": 1, "y": 2}, sense="<=", rhs=3, name="lin_leq")
mod.linear_constraint(linear={"x": 1, "y": 2}, sense=">=", rhs=3, name="lin_geq")
print(mod.prettyprint())

# Add quadratic constraints
mod.quadratic_constraint(
    linear={"x": 1, "y": 1},
    quadratic={("x", "x"): 1, ("y", "z"): -1},
    sense="==",
    rhs=1,
    name="quad_eq",
)
mod.quadratic_constraint(
    linear={"x": 1, "y": 1},
    quadratic={("x", "x"): 1, ("y", "z"): -1},
    sense="<=",
    rhs=1,
    name="quad_leq",
)
mod.quadratic_constraint(
    linear={"x": 1, "y": 1},
    quadratic={("x", "x"): 1, ("y", "z"): -1},
    sense=">=",
    rhs=1,
    name="quad_geq",
)
print(mod.prettyprint())

#Access linear and quadratic terms of linear and quadraic constraints in the same way as the objective funciton
lin_geq = mod.get_linear_constraint("lin_geq")
print("lin_geq:", lin_geq.linear.to_dict(use_name=True), lin_geq.sense, lin_geq.rhs)
quad_geq = mod.get_quadratic_constraint("quad_geq")
print(
    "quad_geq:",
    quad_geq.linear.to_dict(use_name=True),
    quad_geq.quadratic.to_dict(use_name=True),
    quad_geq.sense,
    lin_geq.rhs,
)

# Remove constraints
mod.remove_linear_constraint("lin_eq")
mod.remove_quadratic_constraint("quad_leq")
print(mod.prettyprint())

#Substituting variables
sub = mod.substitute_variables(constants={"x": 0}, variables={"y": ("z", -1)})
print(sub.prettyprint())

## Convert a given optimization problem to a new problem that is a QUBO
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

problem = QuadraticProgram()
#define a problem
conv = QuadraticProgramToQubo()
problem2 = conv.convert(problem)
