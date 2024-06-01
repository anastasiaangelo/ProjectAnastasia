import numpy as np
from copy import deepcopy
import pandas as pd
import ast
import json

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.primitives import Estimator
from qiskit import QuantumCircuit, transpile

import Ising_Hamiltonian_production as IHP

def get_hamiltonian(num_rot, num_res):
    IHP.create_energy_files(num_res, num_rot)
    file_path = f"RESULTS/XY-QAOA/{num_res}res-{num_rot}rot.csv"

    df1 = pd.read_csv(f"energy_files/{num_rot}rot_{num_res}res_one_body_terms.csv")
    q = df1['E_ii'].values
    num = len(q)
    N = int(num/num_rot)
    num_qubits = num

    df = pd.read_csv(f"energy_files/{num_rot}rot_{num_res}res_two_body_terms.csv")
    value = df['E_ij'].values
    Q = np.zeros((num,num))
    n = 0

    if num_rot == 2:
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

    
    elif num_rot == 3:

        for j in range(0, num-3, num_rot):
            for i in range(j, j+num_rot):
                Q[i][j+3] = deepcopy(value[n])
                Q[j+3][i] = deepcopy(value[n])
                Q[i][j+4] = deepcopy(value[n+1])
                Q[j+4][i] = deepcopy(value[n+1])
                Q[i][j+5] = deepcopy(value[n+2])
                Q[j+5][i] = deepcopy(value[n+2])
                n += num_rot

    else:    
        raise ValueError("Number of rotomers not supported.")
    
    H = np.zeros((num,num))

    for i in range(num):
        for j in range(num):
            if i != j:
                H[i][j] = np.multiply(0.25, Q[i][j])

    for i in range(num):
        H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

    return H

def X_op(i, num_qubits):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = ['I'] * num_qubits
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))

def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
        
    pauli_str = ['I']*n

    if i == j:
        pauli_str[i] = 'Z'
    else:
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))

def get_XY_mixer(num_qubits, num_rot):
    if num_rot == 2:
        hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])  
        for i in range(0, num_qubits, 2):
            if i + 1 < num_qubits:
                xx_term = ['I'] * num_qubits
                yy_term = ['I'] * num_qubits
                xx_term[i] = 'X'
                xx_term[i+1] = 'X'
                yy_term[i] = 'Y'
                yy_term[i+1] = 'Y'
                xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
                yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])
                hamiltonian += xx_op + yy_op
        return -hamiltonian 
    elif num_rot == 3:
        hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
        for i in range(0, num_qubits - 2, 3): 
            x1x2 = ['I'] * num_qubits
            y1y2 = ['I'] * num_qubits
            x2x3 = ['I'] * num_qubits
            y2y3 = ['I'] * num_qubits
            x1x3 = ['I'] * num_qubits
            y1y3 = ['I'] * num_qubits

            x1x2[i] = 'X'
            x1x2[i+1] = 'X'
            y1y2[i] = 'Y'
            y1y2[i+1] = 'Y'
            x2x3[i+1] = 'X'
            x2x3[i+2] = 'X'
            y2y3[i+1] = 'Y'
            y2y3[i+2] = 'Y'
            x1x3[i] = 'X'
            x1x3[i+2] = 'X'
            y1y3[i] = 'Y'
            y1y3[i+2] = 'Y' 

            x1x2 = SparsePauliOp(Pauli(''.join(x1x2)), coeffs=[1/2])
            y1y2 = SparsePauliOp(Pauli(''.join(y1y2)), coeffs=[1/2])
            x2x3 = SparsePauliOp(Pauli(''.join(x2x3)), coeffs=[1/2])
            y2y3 = SparsePauliOp(Pauli(''.join(y2y3)), coeffs=[1/2])
            x1x3 = SparsePauliOp(Pauli(''.join(x1x3)), coeffs=[1/2])
            y1y3 = SparsePauliOp(Pauli(''.join(y1y3)), coeffs=[1/2])

            hamiltonian += x1x2 + y1y2 + x2x3 + y2y3 + x1x3 + y1y3
        return hamiltonian
    else:
        raise ValueError("Number of rotomers not supported.")
    
def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
        
    pauli_str = ['I']*n

    if i == j:
        pauli_str[i] = 'Z'
    else:
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))

def get_q_hamiltonian(num_qubits, H):
    q_hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if H[i][j] != 0:
                pauli = generate_pauli_zij(num_qubits, i, j)
                op = SparsePauliOp(pauli, coeffs=[H[i][j]])
                q_hamiltonian += op

    for i in range(num_qubits):
        pauli = generate_pauli_zij(num_qubits, i, i)
        Z_i = SparsePauliOp(pauli, coeffs=[H[i][i]])
        q_hamiltonian += Z_i

    return q_hamiltonian
    
def generate_initial_bitstring(num_qubits, num_rot):
    if num_rot == 2:
        bitstring = [(i%2) for i in range(num_qubits)]
        return ''.join(map(str, bitstring))
    elif num_rot == 3:
        pattern = '100'
        bitstring = (pattern * (num_qubits // 3 + 1))[:num_qubits]
        return bitstring
    else:
        raise ValueError("Number of rotomers not supported.")
    
def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)



def int_to_bitstring(state, total_bits):
    """Converts an integer state to a binary bitstring with padding of leading zeros."""
    return format(state, '0{}b'.format(total_bits))

def check_hamming(bitstring, substring_size):
    """Check if each substring contains exactly one '1'."""
    substrings = [bitstring[i:i+substring_size] for i in range(0, len(bitstring), substring_size)]
    return all(sub.count('1') == 1 for sub in substrings)

def calculate_bitstring_energy(bitstring, hamiltonian, backend=None):
    """
    Calculate the energy of a given bitstring for a specified Hamiltonian.

    Args:
        bitstring (str): The bitstring for which to calculate the energy.
        hamiltonian (SparsePauliOp): The Hamiltonian operator of the system, defined as a SparsePauliOp.
        backend (qiskit.providers.Backend): The quantum backend to execute circuits.

    Returns:
        float: The calculated energy of the bitstring.
    """
    # Prepare the quantum circuit for the bitstring
    num_qubits = len(bitstring)
    qc = QuantumCircuit(num_qubits)
    for i, char in enumerate(bitstring):
        if char == '1':
            qc.x(i)  # Apply X gate if the bit in the bitstring is 1
    
    # Use Aer's statevector simulator if no backend provided
    if backend is None:
        backend = Aer.get_backend('aer_simulator_statevector')

    qc = transpile(qc, backend)
    estimator = Estimator()
    resultt = estimator.run(observables=[hamiltonian], circuits=[qc], backend=backend).result()

    return resultt.values[0].real

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating string: {value}, {e}")
        return None
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)