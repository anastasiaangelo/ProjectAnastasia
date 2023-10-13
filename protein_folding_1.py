# First script using qiski_research.protein_folding to model protein folding in qiskit using configuration and interaction qubits
from qiskit_research.protein_folding.interactions.random_interaction import (
    RandomInteraction,
)
from qiskit_research.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_research.protein_folding.peptide.peptide import Peptide
from qiskit_research.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)

from qiskit_research.protein_folding.penalty_parameters import PenaltyParameters

from qiskit.utils import algorithm_globals, QuantumInstance

algorithm_globals.random_seed = 23

# we demonstrate the generation of the qubit operator in a neuropeptide with the main chain consisting of 7 aminoacids with letter codes APRLRFY 
main_chain = "APRLRFY"

#Our model allows for side chains of the maximum length of one
side_chains = [""] * 7

# For the description of inter-residue contacts for proteins we use knowledge-based (statistical) potentials derived using quasi-chemical approximation
random_interaction = RandomInteraction()
mj_interaction = MiyazawaJerniganInteraction()

# To ensure that all physical constraints are respected we introduce penalty functions
penalty_back = 10       #to penalize turns along the same axis. This term is used to eliminate sequences where the same axis is chosen twice in a row. In this way we do not allow for a chain to fold back into itself.
penalty_chiral = 10     #to impose the right chirality.
penalty_1 = 10          #to penalize local overlap between beads within a nearest neighbor contact.

penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)

#we define the peptide object that includes all the structural information of the modeled system.
peptide = Peptide(main_chain, side_chains)

#Based on the defined peptide, the interaction (contact map) and the penalty terms we defined for our model we define the protein folding problem that returns qubit operators.
protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
qubit_op = protein_folding_problem.qubit_op()
print(qubit_op)

## Using VQE with CVaR expectation value for the solution of the problem
# we are targeting the single bitstring that gives us the minimum energy (corresponding to the folded structure of the protein). 
# Thus, we can use the Variational Quantum Eigensolver with Conditional Value at Risk (CVaR) expectation values for the solution of the problem and for finding the minimum configuration energy
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit import execute, Aer
from qiskit.primitives import Sampler

# set classical optimizer
optimizer = COBYLA(maxiter=50)      #used to optimize the parameters of the quantum circuit to find the minimum eigenvalue.

# set variational ansatz using RealAmplitudes with a specified number of repetitions (reps=1)
ansatz = RealAmplitudes(reps=1)     #ansatz is the parameterized quantum circuit used to prepare trial states and varies to minimize the energy.

counts = []
values = []

def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

# initialize VQE using CVaR with alpha = 0.1
vqe = SamplingVQE(
    Sampler(),
    ansatz=ansatz,
    optimizer=optimizer,
    aggregation=0.1,
    callback=store_intermediate_result,
)

raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
print(raw_result)

#visualise the result
import matplotlib.pyplot as plt

fig = plt.figure()

plt.plot(counts, values)
plt.ylabel("Conformation Energy")
plt.xlabel("VQE Iterations")

fig.add_axes([0.44, 0.51, 0.44, 0.32])

plt.plot(counts[40:], values[40:])
plt.ylabel("Conformation Energy")
plt.xlabel("VQE Iterations")
plt.show()

#The shape of the protein has been encoded by a sequence of turns , . Each turn represents a different direction in the lattice.
#For a main bead of Naa in a lattice, we need Naa-1 turns in order to represent its shape. However, the orientation of the protein is not relevant to its energy

result = protein_folding_problem.interpret(raw_result=raw_result)
print(
    "The bitstring representing the shape of the protein during optimization is: ",
    result.turn_sequence,
)
print("The expanded expression is:", result.get_result_binary_vector())

#Now that we know which qubits encode which information, we can decode the bitstring into the explicit turns that form the shape of the protein.
print(
    f"The folded protein's main sequence of turns is: {result.protein_shape_decoder.main_turns}"
)
print(f"and the side turn sequences are: {result.protein_shape_decoder.side_turns}")

#From this sequence of turns we can get the cartesian coordinates of each of the aminoacids of the protein.
print(result.protein_shape_file_gen.get_xyz_data())

#And finally, we can also plot the structure of the protein in 3D. Note that when rendered with the proper backend this plot can be interactively rotated.
fig = result.get_figure(title="Protein Structure", ticks=False, grid=True)
fig.get_axes()[0].view_init(10, 70)
fig.savefig("protein_structure.png")


## An example with sidechains


