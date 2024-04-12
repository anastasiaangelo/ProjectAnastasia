# Hybrid model to encode 2 rotamers per qubit, to reduce circuit depth, check for 4 qubits per reisdue, so 8 rotamers per residue
# %%
import numpy as np
import pandas as pd

qubit_per_res = 4

num_rot = 2*qubit_per_res

df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N_res = int(num/num_rot)

df = pd.read_csv("energy_files/two_body_terms.csv")
v = df['E_ij'].values
numm = len(v)

print("q: \n", q)

num_qubits = N_res * qubit_per_res

# %%
## Quantum optimisation
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


i = 0 #each loop is one residue
for _ in range(N_res): #rediue i
    for j in range(0, num_qubits):   #each qubit for residue i
        N_0j = N_0(j, num_qubits)
        N_1j = N_1(j, num_qubits)
        if i >= num:
            break 
        H_self += q[i] * N_0j + q[i+1] * N_1j
        i += 2
     


i = 0 #one loop is one pair of residues
for _ in range(N_res):    # residue i
    for k in range(0, num_qubits):       #qubits 0-3 for residue i
        for l in range(0, num_qubits):   #qubits 0-3 for residue j
            N_0k = N_0(k, num_qubits)
            N_1k = N_1(k, num_qubits)
            N_0l = N_0(l, num_qubits)
            N_1l = N_1(l, num_qubits)
            if i >= numm:
                break  
            H_int += v[i] * N_0k @ N_0l + v[i+1] * N_0k @ N_1l + v[i+2] * N_1k @ N_0l + v[i+3] * N_1k @ N_1l
            i += num_qubits

H_gen = H_int + H_self

def add_hamming_penalty(H, num_qubits, qubits_per_res, P):
    for res_start in range(0, num_qubits, qubits_per_res):  # iterate over all residues
        for i in range(res_start, res_start + qubits_per_res):         # For each residue, add a penalty for every pair of qubits
            for j in range(i + 1, res_start + qubits_per_res):
                # Create Z operators for the pair
                z_i = SparsePauliOp(Pauli(('I' * i + 'Z' + 'I' * (num_qubits - i - 1))), coeffs=[P/2])
                z_j = SparsePauliOp(Pauli(('I' * j + 'Z' + 'I' * (num_qubits - j - 1))), coeffs=[P/2])
                # Add penalty term for the pair to the Hamiltonian
                H += z_i @ z_j
    return H

# Choose a suitable penalty constant, ensuring it's significant compared to your energy scales
P = 10

H_gen = add_hamming_penalty(H_gen, num_qubits, qubit_per_res, P)

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
# %%

# from qiskit_aer.noise import NoiseModel
# from qiskit_ibm_provider import IBMProvider
# from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session, Sampler

# IBMProvider.save_account('25a4f69c2395dfbc9990a6261b523fe99e820aa498647f92552992afb1bd6b0bbfcada97ec31a81a221c16be85104beb653845e23eeac2fe4c0cb435ec7fc6b4', overwrite=True)
# provider = IBMProvider()
# available_backends = provider.backends()
# print([backend.name for backend in available_backends])
# service = QiskitRuntimeService(channel="ibm_quantum")
# backend = service.backend("ibmq_qasm_simulator")
# noise_model = NoiseModel.from_backend(backend)
# simulator = AerSimulator(noise_model = noise_model)
# print('Noise model', noise_model)

# options = Options()
# options.simulator = {
#     "noise_model":  noise_model,
#     "basis_gates": backend.configuration().basis_gates,
#     "coupling_map": backend.configuration().coupling_map,
#     "seed_simulator": 42
# }
# options.execution.shots = 1000
# options.optimization_level = 0
# options.resilience_level = 0

# with Session(service=service, backend=backend):
#     sampler = Sampler(options=options)
#     qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
#     result1 = qaoa1.compute_minimum_eigenvalue(H_gen)
#     print('Running noisy simulation..')

# print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
# print('best measurement', result1.best_measurement)
# print('Optimal parameters: ', result1.optimal_parameters)
# print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
#%%

