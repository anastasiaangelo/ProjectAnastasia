# %%
import numpy as np
import pandas as pd
import itertools
import functools
import operator
from itertools import combinations
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qubit_per_res = 2
num_rot = 2**qubit_per_res

df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("energy_files/two_body_terms.csv")
v = df['E_ij'].values
numm = len(v)

print("q: \n", q)

num_qubits = N_res * qubit_per_res

## Quantum optimisation
from qiskit_aer import Aer
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

## Mapping to qubits
H_self = SparsePauliOp(Pauli('I'* num_qubits), coeffs=[0])
H_int = SparsePauliOp(Pauli('I'* num_qubits), coeffs=[0]) 

def N_0(i, n):
    pauli_str = ['I'] * n
    pauli_str[i] = 'Z'
    z_op = SparsePauliOp(Pauli(''.join(pauli_str)), coeffs=[0.5])
    i_op = SparsePauliOp(Pauli('I'*n), coeffs=[0.5])
    return z_op + i_op

def N_1(i, n):
    pauli_str = ['I'] * n
    pauli_str[i] = 'Z'
    z_op = SparsePauliOp(Pauli(''.join(pauli_str)), coeffs=[-0.5])
    i_op = SparsePauliOp(Pauli('I'*n), coeffs=[0.5])
    return z_op + i_op


def create_pauli_operators(num_qubits, qubits_per_res):
    operators = []
    for i in range(0, num_qubits, qubits_per_res):
        # Generate all combinations of N_0 and N_1 for the residue
        for comb in itertools.product([N_0, N_1], repeat=qubits_per_res):
            ops = [func(j, num_qubits) for j, func in enumerate(comb, start=i)]
            # Now you have a list of operators for each qubit in the residue
            full_op = functools.reduce(operator.matmul, ops)
            operators.append(full_op)
    return operators

for i, op in enumerate(create_pauli_operators(num_qubits, qubit_per_res)):
    H_self += q[i] * op
    if i >= len(q) - 1:
        break

def create_interaction_operators(num_qubits, qubits_per_res, v):
    H_int = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
    v_index = 0
    
    # Iterate over all unique pairs of residues
    for res1, res2 in combinations(range(0, num_qubits, qubits_per_res), 2):
        # Generate all combinations of N_0 and N_1 for each qubit in the residue
        for comb1 in itertools.product([N_0, N_1], repeat=qubits_per_res):
            for comb2 in itertools.product([N_0, N_1], repeat=qubits_per_res):
                op_list1 = [func(i + res1, num_qubits) for i, func in enumerate(comb1)]
                op_list2 = [func(j + res2, num_qubits) for j, func in enumerate(comb2)]
                
                # Now you have two lists of operators, one for each residue in the pair
                full_op1 = functools.reduce(operator.matmul, op_list1)
                full_op2 = functools.reduce(operator.matmul, op_list2)
                
                H_int += v[v_index] * full_op1 @ full_op2
                v_index += 1
                
                if v_index >= len(v) - 1:
                    break
            if v_index >= len(v) - 1:
                break
        if v_index >= len(v) - 1:
            break

    return H_int

H_int = create_interaction_operators(num_qubits, qubit_per_res, v)

H_gen = H_self + H_int


def X_op(i, num):
    op_list = ['I'] * num
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))

mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
p = 1
initial_point = np.ones(2*p)
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result_gen = qaoa.compute_minimum_eigenvalue(H_gen)
print("\n\nThe result of the quantum optimisation using QAOA is: \n")
print('best measurement', result_gen.best_measurement)
print('The ground state energy with QAOA is: ', np.real(result_gen.best_measurement['value']))

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeKolkataV2


# # Fake backends for local simulations
# fake_backend = FakeKolkataV2()
# pm = generate_preset_pass_manager(backend=fake_backend, optimization_level=1)

# ansatz = QAOAAnsatz(H_gen, reps=2)
# ansatz_isa =pm.run(ansatz)
# ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")

# hamiltonian_isa = H_gen.apply_layout(ansatz_isa.layout)

# options = {'simulator':{'seed_simulator': 42}}
# sampler = Sampler(backend=fake_backend, options=options)

# result = sampler.run([hamiltonian_isa]).result()
# print('fakebackend local simulation:', result)

# num_parameters = ansatz_isa.num_parameters
# print(f"Number of parameters in the modified ansatz: {num_parameters}")
# initial_point_isa = np.ones(2)
# x0 = 2 * np.pi * np.random.rand(ansatz_isa.num_parameters)

# qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=x0)
# result1 = qaoa1.compute_minimum_eigenvalue(hamiltonian_isa)
# print('Running noisy simulation..')


# %%
# AerSimulator for local simulations
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

qc = QuantumCircuit(H_gen.num_qubits)

print(qc.draw())

aer_sim = AerSimulator()
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=3)
hamiltonian_isa = pm.run(qc)
  

# ansatz = QAOAAnsatz(H_gen, reps=2)
# ansatz_isa =pm.run(ansatz)
# ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")

# hamiltonian_isa = QuantumCircuit(num_qubits,num_qubits)
# hamiltonian_isa.append(Operator(H_gen).to_instruction(),range(num_qubits))
# hamiltonian_isa = pm.run(hamiltonian_isa)
# # for q in range(0,num_qubits-1):
# #     hamiltonian_isa.rxx(np.pi/5,q,q+1)
# # hamiltonian_isa.measure_all()
# hamiltonian_isa.draw()

sampler = Sampler()
result = sampler.run([hamiltonian_isa]).result()    
print('aer simulation results: ', result)

#%%
backend = Aer.get_backend('aer_simulator')
qaoa1 = QAOA(sampler= Sampler(), optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
result1 = qaoa1.compute_minimum_eigenvalue(H_gen)
print("Ground state energy:", result.eigenvalue.real)
print("Ground state:", result.eigenstate)


#%%

# # To run on local simulator
# from qiskit.primitives import StatevectorEstimator
# estimator = StatevectorEstimator()

# x0 = 2 * np.pi * np.random.rand(ansatz_isa.num_parameters)
# res = minimize(cost_func, x0, args=(ansatz_isa, hamiltonian_isa, estimator), method="COBYLA")
# print('res: ', res)

## as before
# options = Options()
# options.simulator = {
#     "noise_model":  noise_model,
#     "basis_gates": backend.configuration().basis_gates,
#     "seed_simulator": 42
# }
# options.execution.shots = 1000
# options.optimization_level = 0
# options.resilience_level = 0

# with Session(service=service, backend=backend) as session:
#     # sampler = Sampler(options=options)
#     sampler = Sampler(session=session)
#     qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point_isa)
#     result1 = qaoa1.compute_minimum_eigenvalue(hamiltonian_isa)
#     print('Running noisy simulation..')

# print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
# print('best measurement', result1.best_measurement)
# print('Optimal parameters: ', result1.optimal_parameters)
# print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))



# %%
