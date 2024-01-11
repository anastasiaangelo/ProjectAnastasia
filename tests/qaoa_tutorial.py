#qaoa tutorial for the maxcut problem
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
nx.draw(G, with_labels=True, alpha=0.8, node_size=500)
plt.show()

# The mixing unitary

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter

# Adjacency is essentially a matrix which tells you which nodes are
# connected. This matrix is given as a sparse matrix, so we need to
# convert it to a dense matrix
adjacency = nx.adjacency_matrix(G).todense()

nqubits = 4

beta = Parameter("$\\beta$")
qc_mix = QuantumCircuit(nqubits)
for i in range(0, nqubits):
    qc_mix.rx(2*beta, i)

qc_mix.draw()
print(qc_mix)

# The problem unitary

gamma = Parameter("$\\gamma$")
qc_p = QuantumCircuit(nqubits)
for pair in list(G.edges()): #pairs of ndoes
    qc_p.rzz(2*gamma, pair[0], pair[1])
    qc_p.barrier()

qc_p.decompose().draw()
print(qc_p.decompose())

# The initial state - Such a state, can be prepared by applying Hadamard gates starting from an all zero state as shown in the circuit below.

qc_0 = QuantumCircuit(nqubits)
for i in range(0, nqubits):
    qc_0.h(i)

print(qc_0)

# The QAOA cirucit
# the preparation of a quantum state during qaoa is composed of three elements: preparing an initial state, applting the unitary U(Hp) corresponfing to the problem hamiltonian, then applying the mixing unitary U(Hb)

qc_qaoa = QuantumCircuit(nqubits)

qc_qaoa.append(qc_0, [i for i in range(0, nqubits)])
qc_qaoa.append(qc_mix, [i for i in range(0,nqubits)])
qc_qaoa.append(qc_p, [i for i in range(0, nqubits)])

qc_qaoa.decompose().decompose().draw()
print(qc_qaoa.decompose().decompose())

# the next step is to find the optimal parameters (betaopt, gammaopt) such that the expectation value <psi|Hp|psi(opt)> is minimised 
# such an expectation can be obtained by doing measurement in the Z basis. We use a classical optimisation algorithm to find the optimal parameters

def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.

    Args:
        x: str
           solution bitstring

        G: networkx graph

    Returns:
        obj: float
             Objective
    """
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1

    return obj



def compute_expectation(counts, G):
    """
    Computes expectation value based on measurement results

    Args:
        counts: dict
                key as bitstring, val as count

        G: networkx graph

    Returns:
        avg: float
             expectation value
    """

    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():

        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count

    return avg/sum_count


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(G, theta):
    """
    Creates a parametrized qaoa circuit

    Args:
        G: networkx graph
        theta: list
               unitary parameters

    Returns:
        qc: qiskit circuit
    """

    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):

        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc


from qiskit_rigetti import RigettiQCSProvider
provider = RigettiQCSProvider()

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(G, p, shots=512):
    """
    Runs parametrized circuit

    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    backend = provider.get_simulator(num_qubits=len(G.nodes), noisy=False)  # or provider.get_backend(name='Aspen-9')

    def execute_circ(theta):

        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, shots=512).result().get_counts()

        return compute_expectation(counts, G)

    return execute_circ


from scipy.optimize import minimize

expectation = get_expectation(G, p=1)

res = minimize(expectation,
                      [1.0, 1.0],
                      method='COBYLA')
res


# Analysing the result

from qiskit.visualization import plot_histogram

backend = provider.get_simulator(num_qubits = len(G.nodes), noisy = False)
qc_res = create_qaoa_circ(G, res.x)
counts = backend.run(qc_res, shots=512).result().get_counts()
plot_histogram(counts)
plt.show()