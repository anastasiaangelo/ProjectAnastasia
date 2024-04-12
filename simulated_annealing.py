# Script to optimise the Hamiltonian, starting directly from the Ising Hamiltonian
# or build the Pauli representation from the problem may be more efficient rather than converting it
# too complex though for now 
import numpy as np
import pandas as pd
from copy import deepcopy

num_rot = 2

## configure the hamiltonian from the values calculated classically with pyrosetta
df1 = pd.read_csv("energy_files/one_body_terms.csv")
q = df1['E_ii'].values
num = len(q)
N = int(num/num_rot)

print('Qii values: \n', q)

df = pd.read_csv("energy_files/two_body_terms.csv")
value = df['E_ij'].values
Q = np.zeros((num,num))
n = 0

for i in range(0, num-2):
    if i%2 == 0:
        Q[i][i+2] = deepcopy(value[n])
        Q[i+2][i] = deepcopy(value[n])
        Q[i][i+3] = deepcopy(value[n+1])
        Q[i+3][i] = deepcopy(value[n+1])
        n += 2
    elif i%2 != 0:
        Q[i][i+1] = deepcopy(value[n])
        Q[i+1][i] = deepcopy(value[n])
        Q[i][i+2] = deepcopy(value[n+1])
        Q[i+2][i] = deepcopy(value[n+1])
        n += 2

print('\nQij values: \n', Q)

H = np.zeros((num,num))

for i in range(num):
    for j in range(num):
        if i != j:
            H[i][j] = np.multiply(0.25, Q[i][j])

for i in range(num):
    H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

print('\nH: \n', H)

# add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
def add_penalty_term(M, penalty_constant, residue_pairs):
    for i, j in residue_pairs:
        M[i][j] += penalty_constant
        
    return M

P = 6

def generate_pairs(N):
    pairs = [(i, i+1) for i in range(0, 2*N, 2)]
    return pairs

pairs = generate_pairs(N)

M = deepcopy(H)
M = add_penalty_term(M, P, pairs)

## Classical optimisation:
from scipy.sparse.linalg import eigsh
from scipy.optimize import basinhopping

num_qubits = num

Z_matrix = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

def construct_operator(qubit_indices, num_qubits):
    operator = np.eye(1)
    for qubit in range(num_qubits):
        if qubit in qubit_indices:
            operator = np.kron(operator, Z_matrix)
        else:
            operator = np.kron(operator, identity)
    return operator

C = np.zeros((2**num_qubits, 2**num_qubits))

for i in range(num_qubits):
    operator = construct_operator([i], num_qubits)
    C += H[i][i] * operator

for i in range(num_qubits):
    for j in range(i+1, num_qubits):
        operator = construct_operator([i, j], num_qubits)
        C += H[i][j] * operator

print('C :\n', C)

def create_hamiltonian(pairs, P, num_qubits):
    H_pen = np.zeros((2**num_qubits, 2**num_qubits))
    def tensor_term(term_indices):
        term = [Z_matrix if i in term_indices else identity for i in range(num_qubits)]
        result = term[0]
        for t in term[1:]:
            result = np.kron(result, t)
        return result
    
    for pair in pairs:
        term = tensor_term(pair)
        H_pen += P * term

    return H_pen

H_penalty = create_hamiltonian(pairs, P, num_qubits)
H_tot = C + H_penalty

# Extract the ground state energy and wavefunction
# using sparse representation so as to be able to generalise to larger systems
eigenvalues, eigenvectors = eigsh(H_tot, k=num, which='SA')
print("\n\nClassical optimisation results. \n")
print("Ground energy eigsh: ", eigenvalues[0])
print("ground state wavefuncion eigsh: ", eigenvectors[:,0])

# Simulated Annealing minimisation benchmark
# Initial guess for the parameters
x0 = [0.5, 0.5, 0.5]  # Adjust based on your actual number of parameters and their expected range

# Define the step size for making jumps
class MyTakeStep:
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        x += np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
        return x

# Configure the basinhopping algorithm
minimizer_kwargs = {"method": "BFGS"}  # Local minimization algorithm
take_step = MyTakeStep(stepsize=0.5)

result = basinhopping(H_tot, x0, minimizer_kwargs=minimizer_kwargs, niter=100, T=1.0, stepsize=0.5, take_step=take_step)
print("Global minimum with SA: ", result.fun)
print("Parameters at minimum: ", result.x)

