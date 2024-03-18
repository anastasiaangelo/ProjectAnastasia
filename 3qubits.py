import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
import cmath

qubit_per_res = 3
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
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

## Mapping to qubits
# for 3 qubits per residue, 4 rotamers per residue
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
for j in range(0, num_qubits, qubit_per_res):
    N_0i = N_0(j, num_qubits)
    N_1i = N_1(j, num_qubits)
    N_0j = N_0(j+1, num_qubits)
    N_1j = N_1(j+1, num_qubits)
    N_0k = N_0(j+2, num_qubits)
    N_1k = N_1(j+2, num_qubits)
    H_self += q[i] * N_0i @ N_0j @ N_0k + q[i+1] * N_0i @ N_0j @ N_1k + q[i+2] * N_0i @ N_1j @ N_0k + q[i+3] * N_0i @ N_1j @ N_1k + q[i+4] * N_1i @ N_0j @ N_0k + q[i+5] * N_1i @ N_0j @ N_1k + q[i+6] * N_1i @ N_1j @ N_0k + q[i+7] * N_1i @ N_1j @ N_1k
    i += num_rot
    if i >= num:
        break


i = 0     #one loop is one pair of residues
for j in range(0, num_qubits, qubit_per_res):
    for k in range(qubit_per_res, num_qubits, qubit_per_res):
        N_0i = N_0(j, num_qubits)
        N_1i = N_1(j, num_qubits)
        N_0ii = N_0(j+1, num_qubits)
        N_1ii = N_1(j+1, num_qubits)
        N_0iii = N_0(j+2, num_qubits)
        N_1iii = N_1(j+2, num_qubits)
        N_0j = N_0(k, num_qubits)
        N_1j = N_1(k, num_qubits)
        N_0jj = N_0(k+1, num_qubits)
        N_1jj = N_1(k+1, num_qubits)
        N_0jjj = N_0(k+2, num_qubits)
        N_1jjj = N_1(k+2, num_qubits)
        if i >= numm:
            break
        H_int += v[i] * N_0i @ N_0ii @ N_0iii @ N_0j @ N_0jj @ N_0jjj + v[i+1] * N_0i @ N_0ii @ N_0iii @ N_0j @ N_0jj @ N_1jjj + v[i+2] * N_0i @ N_0ii @ N_0iii @ N_0j @ N_1jj @ N_0jjj + v[i+3] * N_0i @ N_0ii @ N_0iii @ N_0j @ N_1jj @ N_1jjj + v[i+4] * N_0i @ N_0ii @ N_0iii @ N_1j @ N_0jj @ N_0jjj + v[i+5] * N_0i @ N_0ii @ N_0iii @ N_1j @ N_0jj @ N_1jjj + v[i+6] * N_0i @ N_0ii @ N_0iii @ N_1j @ N_1jj @ N_0jjj + v[i+7] * N_0i @ N_0ii @ N_0iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+8] * N_0i @ N_0ii @ N_1iii @ N_0j @ N_0jj @ N_0jjj + v[i+9] * N_0i @ N_0ii @ N_1iii @ N_0j @ N_0jj @ N_1jjj + v[i+10] * N_0i @ N_0ii @ N_1iii @ N_0j @ N_1jj @ N_0jjj + v[i+11] * N_0i @ N_0ii @ N_1iii @ N_0j @ N_1jj @ N_1jjj + v[i+12] * N_0i @ N_0ii @ N_1iii @ N_1j @ N_0jj @ N_0jjj + v[i+13] * N_0i @ N_0ii @ N_1iii @ N_1j @ N_0jj @ N_1jjj + v[i+14] * N_0i @ N_0ii @ N_1iii @ N_1j @ N_1jj @ N_0jjj + v[i+15] * N_0i @ N_0ii @ N_1iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+16] * N_0i @ N_1ii @ N_0iii @ N_0j @ N_0jj @ N_0jjj + v[i+17] * N_0i @ N_1ii @ N_0iii @ N_0j @ N_0jj @ N_1jjj + v[i+18] * N_0i @ N_1ii @ N_0iii @ N_0j @ N_1jj @ N_0jjj + v[i+19] * N_0i @ N_1ii @ N_0iii @ N_0j @ N_1jj @ N_1jjj + v[i+20] * N_0i @ N_1ii @ N_0iii @ N_1j @ N_0jj @ N_0jjj + v[i+21] * N_0i @ N_1ii @ N_0iii @ N_1j @ N_0jj @ N_1jjj + v[i+22] * N_0i @ N_1ii @ N_0iii @ N_1j @ N_1jj @ N_0jjj + v[i+23] * N_0i @ N_1ii @ N_0iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+24] * N_0i @ N_1ii @ N_1iii @ N_0j @ N_0jj @ N_0jjj + v[i+25] * N_0i @ N_1ii @ N_1iii @ N_0j @ N_0jj @ N_1jjj + v[i+26] * N_0i @ N_1ii @ N_1iii @ N_0j @ N_1jj @ N_0jjj + v[i+27] * N_0i @ N_1ii @ N_1iii @ N_0j @ N_1jj @ N_1jjj + v[i+28] * N_0i @ N_1ii @ N_1iii @ N_1j @ N_0jj @ N_0jjj + v[i+29] * N_0i @ N_1ii @ N_1iii @ N_1j @ N_0jj @ N_1jjj + v[i+30] * N_0i @ N_1ii @ N_1iii @ N_1j @ N_1jj @ N_0jjj + v[i+31] * N_0i @ N_1ii @ N_1iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+32] * N_1i @ N_0ii @ N_0iii @ N_0j @ N_0jj @ N_0jjj + v[i+33] * N_1i @ N_0ii @ N_0iii @ N_0j @ N_0jj @ N_1jjj + v[i+34] * N_1i @ N_0ii @ N_0iii @ N_0j @ N_1jj @ N_0jjj + v[i+35] * N_1i @ N_0ii @ N_0iii @ N_0j @ N_1jj @ N_1jjj + v[i+36] * N_1i @ N_0ii @ N_0iii @ N_1j @ N_0jj @ N_0jjj + v[i+37] * N_1i @ N_0ii @ N_0iii @ N_1j @ N_0jj @ N_1jjj + v[i+38] * N_1i @ N_0ii @ N_0iii @ N_1j @ N_1jj @ N_0jjj + v[i+39] * N_1i @ N_0ii @ N_0iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+40] * N_1i @ N_0ii @ N_1iii @ N_0j @ N_0jj @ N_0jjj + v[i+41] * N_1i @ N_0ii @ N_1iii @ N_0j @ N_0jj @ N_1jjj + v[i+42] * N_1i @ N_0ii @ N_1iii @ N_0j @ N_1jj @ N_0jjj + v[i+43] * N_1i @ N_0ii @ N_1iii @ N_0j @ N_1jj @ N_1jjj + v[i+44] * N_1i @ N_0ii @ N_1iii @ N_1j @ N_0jj @ N_0jjj + v[i+45] * N_1i @ N_0ii @ N_1iii @ N_1j @ N_0jj @ N_1jjj + v[i+46] * N_1i @ N_0ii @ N_1iii @ N_1j @ N_1jj @ N_0jjj + v[i+47] * N_1i @ N_0ii @ N_1iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+48] * N_1i @ N_1ii @ N_0iii @ N_0j @ N_0jj @ N_0jjj + v[i+49] * N_1i @ N_1ii @ N_0iii @ N_0j @ N_0jj @ N_1jjj + v[i+50] * N_1i @ N_1ii @ N_0iii @ N_0j @ N_1jj @ N_0jjj + v[i+51] * N_1i @ N_1ii @ N_0iii @ N_0j @ N_1jj @ N_1jjj + v[i+52] * N_1i @ N_1ii @ N_0iii @ N_1j @ N_0jj @ N_0jjj + v[i+53] * N_1i @ N_1ii @ N_0iii @ N_1j @ N_0jj @ N_1jjj + v[i+54] * N_1i @ N_1ii @ N_0iii @ N_1j @ N_1jj @ N_0jjj + v[i+55] * N_1i @ N_1ii @ N_0iii @ N_1j @ N_1jj @ N_1jjj + \
                + v[i+56] * N_1i @ N_1ii @ N_1iii @ N_0j @ N_0jj @ N_0jjj + v[i+57] * N_1i @ N_1ii @ N_1iii @ N_0j @ N_0jj @ N_1jjj + v[i+58] * N_1i @ N_1ii @ N_1iii @ N_0j @ N_1jj @ N_0jjj + v[i+59] * N_1i @ N_1ii @ N_1iii @ N_0j @ N_1jj @ N_1jjj + v[i+60] * N_1i @ N_1ii @ N_1iii @ N_1j @ N_0jj @ N_0jjj + v[i+61] * N_1i @ N_1ii @ N_1iii @ N_1j @ N_0jj @ N_1jjj + v[i+62] * N_1i @ N_1ii @ N_1iii @ N_1j @ N_1jj @ N_0jjj + v[i+63] * N_1i @ N_1ii @ N_1iii @ N_1j @ N_1jj @ N_1jjj
        
        i += num_rot**2


H_gen = H_int + H_self

# # Visualisation of hamiltonian terms
# # Extract coefficients and terms from the Hamiltonian
# coefficients = []
# terms = []

# for term, coeff in zip(H_gen.paulis, H_gen.coeffs):
#     coefficients.append(coeff)
#     terms.append(str(term))

# # Shorten the list for visualization (if needed)
# # You might need to do this if the number of terms is very large
# coefficients = coefficients[:250]  # adjust this number as needed
# terms = terms[:250]  # adjust this number as needed


# plt.figure(figsize=(10,6))
# plt.bar(terms, coefficients)
# plt.xlabel('Terms')
# plt.ylabel('Coefficients')
# plt.xticks(rotation=90)
# plt.title('Coefficients of Hamiltonian Terms')
# plt.show()

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
print(result_gen)

counts = result_gen.best_measurement
plot_histogram(counts, title="QAOA Measurement Results")

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeKolkata, FakeVigo
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session, Sampler

IBMProvider.save_account('25a4f69c2395dfbc9990a6261b523fe99e820aa498647f92552992afb1bd6b0bbfcada97ec31a81a221c16be85104beb653845e23eeac2fe4c0cb435ec7fc6b4', overwrite=True)
provider = IBMProvider()
available_backends = provider.backends()
print([backend.name for backend in available_backends])
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibmq_qasm_simulator")
noise_model = NoiseModel.from_backend(backend)
simulator = AerSimulator(noise_model = noise_model)
fake_backend = FakeKolkata()
noise_model = NoiseModel.from_backend(fake_backend)
options = Options()
options.simulator = {
    "noise_model": noise_model,
    "basis_gates": fake_backend.configuration().basis_gates,
    "coupling_map": fake_backend.configuration().coupling_map,
    "seed_simulator": 42
}
options.execution.shots = 1000
options.optimization_level = 0
options.resilience_level = 0

with Session(service=service, backend=backend):
    sampler = Sampler(options=options)
    qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
    result1 = qaoa1.compute_minimum_eigenvalue(H_gen)
    print('Running noisy simulation..')

print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
print('best measurement', result1.best_measurement)
print('Optimal parameters: ', result1.optimal_parameters)
print('The ground state energy with noisy QAOA is: ', np.real(result1.best_measurement['value']))
print(result1)



