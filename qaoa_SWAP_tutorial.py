# How to build a QAOA ansatz with SWAP strategies
#the QAOA ansatz is composed of a cost operator and a mic=xer operator applied in an alternating fashion to an initial state
#in this example we fix the initial state and the mixer operator, and show how to build the ansatz for any cost operator, defined as a weighted sum of pauli terms.

import json
import matplotlib.pyplot as plt

graph_file = "graph_2layers_0seed.json"
data = json.load(open(graph_file, "r"))

local_correlators = data["paulis"]
print(local_correlators)

#this list of Paulis can be used to build a sparse operator
from qiskit.quantum_info import SparsePauliOp

cost_operator = SparsePauliOp.from_list(local_correlators)
print(cost_operator)


##Build cirucit for the cost operator Hamiltonian
#The QAOA ansatz is composed of a series of alternating layers of cost operator unitary and mixer unitary blocks. We only want to apply the swap strategies to the cost
#operator layer, so we will start by creating the isolated block that we will later transform and append to the final QAOA circuit.

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz

num_qubits = cost_operator.num_qubits
dummy_initial_state = QuantumCircuit(
    num_qubits
) # the real initial state is defined later
dummy_mixer_operator = QuantumCircuit(num_qubits) # the real mixer is defined later

cost_layer = QAOAAnsatz(
    cost_operator,
    reps=1,
    initial_state=dummy_initial_state,
    mixer_operator=dummy_mixer_operator,
    name="QAOA cost block"
).decompose()
cost_layer.draw("mpl")

cost_layer.decompose(reps=2).draw("mpl")

#if we decompose again we see that these RZZ gates are built by combining RZ and CNOT gates
cost_layer.decompose(reps=3).draw("mpl")

#Once we have defined our cost layer, we will add measurements to all qubits to allow us to keep track of the permutations introduced by the swap strategy
cost_layer.measure_all()

##Apply SWAP Strategies
#we will define a transpiler pass that applies a swap strategy for a given coupling map geometry
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy, FindCommutingPauliEvolutions, Commuting2qGateRouter


# 1 choose swap strategy - in this case a simple linear strategy where each qubit is swapped with the next one
swap_strategy = SwapStrategy.from_line([i for i in range(num_qubits)])
edge_colouring = {(idx, idx+1): (idx+1)% 2 for idx in range(num_qubits)}

# 2 define pass manager - to apply transpiler passes to the circuit before executing it to optimise the circuit for the targer quantum hardware
pm_pre = PassManager(
    [
        FindCommutingPauliEvolutions(), #identifies commuting Pauli gates in the circuit
        Commuting2qGateRouter(          #qubit routing using the swap strategy
            swap_strategy,
            edge_colouring,
        ),
    ]
)

# 3 run transpiler pass to apply swaps
swapped_cost_layer = pm_pre.run(cost_layer) #pass manager is applied
swapped_cost_layer.draw("mpl")

#Build measurement map to recover permutations
#This steps depends on the number of QAOA layers we want our ansatz to contain 

from qopt_best_practices.swap_strategies import make_meas_map

# compute the measurement map (qubit to classical bit) we will apply this for qaoa_layers % 2 == 1
qaoa_layers = 2

#map used to relate qubits to classical bits for measurement purposes
if qaoa_layers % 2 == 1:
    meas_map = make_meas_map(swapped_cost_layer)    
else:
    meas_map = {idx: idx for idx in range(num_qubits)}

# we can now remove the measurements we previously introduced to keep up with the permutations, the measurement map will handle the qubit-to classical-bit mapping and the ealier measurements are no longer needed
swapped_cost_layer.remove_final_measurements()


##Build final QAOA circuit alternating cost layers, permuted cost layers and mixer layers
swapped_cost_layer.parameters

from qiskit.circuit import ParameterVector, Parameter

qaoa_circuit = QuantumCircuit(num_qubits, num_qubits)

#add here initial state, in this case all H
qaoa_circuit.h(range(num_qubits))

#create a gamma and beta parameter per layer
gammas = ParameterVector("γ", qaoa_layers)
betas = ParameterVector("β", qaoa_layers)

#define mixer layer, in this case rx
mixer_layer = QuantumCircuit(num_qubits)
mixer_layer.rx(betas[0], range(num_qubits))

#iterate over number of qaoa layers
for layer in range(qaoa_layers):
    #assign paramaters corresponding to layer
    cost = swapped_cost_layer.assign_parameters(
        {swapped_cost_layer.parameters[0]: gammas[layer]}
    )
    mixer = mixer_layer.assign_parameters({mixer_layer.parameters[0]: betas[layer]})

    if layer % 2 == 0:
        #even layer -> append cost
        qaoa_circuit.append(cost.reverse_ops(), range(num_qubits))
    else: 
        #odd layer -> append reversed cost
        qaoa_circuit.append(cost.reverse_ops(), range(num_qubits))

    #the mixed layer is not reversed
    qaoa_circuit.append(mixer, range(num_qubits))

# to seperate the qaoao layers from the measurement part
qaoa_circuit.barrier()

#iterate over measurement map to recover permutations introduced by swap operations
for qidx, cidx in meas_map.items():
    qaoa_circuit.measure(qidx, cidx)

qaoa_circuit.decompose()
qaoa_circuit.draw("mpl")

## Before exercutiom: parameter binding and transpilation
#how to bind parameters to the circuit and transpile for execution. This is generally done as part of the optimization routine of QAOA.
param_dict = {gammas[0]: 1, gammas[1]: 1, betas[0]: 0, betas[1]: 1}
print(qaoa_circuit.parameters)
final_circuit = qaoa_circuit.bind_parameters(param_dict)

# optional custom transpilation steps go here (to amtch specific hardware)
from qiskit import transpile

basis_gates = ["rz", "sx", "x", "cx"]
#now transpile to sx, rz, x, cx basis
t_final_circuit = transpile(
    final_circuit, basis_gates=basis_gates, optimization_level=2
)
t_final_circuit.draw("mpl")

plt.show()