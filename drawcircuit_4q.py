# from qiskit import QuantumCircuit
# from qiskit.circuit import Gate
# import matplotlib.pyplot as plt

# # Function to create a block of Rz and CNOT gates
# def create_rz_cnot_block(rz_angle, label):
#     qc = QuantumCircuit(2, name=label)
#     qc.rz(rz_angle, 0)
#     qc.cx(0, 1)
#     qc.rz(rz_angle, 1)
#     custom_gate = qc.to_gate()
#     custom_gate.label = label
#     return custom_gate

# # Initialize a 4-qubit quantum circuit
# qc = QuantumCircuit(4)

# # Define angles for Rz rotations
# angles = [1.2, 0.362, 0.455, 0.378, 1.46, 1.17, 0.061, 1.98, 1.81]

# # Apply grouped Rz-CNOT blocks to different pairs of qubits
# qc.append(create_rz_cnot_block(angles[0], "Block_01"), [0, 1])
# qc.append(create_rz_cnot_block(angles[1], "Block_12"), [1, 2])
# qc.append(create_rz_cnot_block(angles[2], "Block_23"), [2, 3])
# qc.append(create_rz_cnot_block(angles[3], "Block_02"), [0, 2])
# qc.append(create_rz_cnot_block(angles[4], "Block_13"), [1, 3])
# qc.append(create_rz_cnot_block(angles[5], "Block_03"), [0, 3])

# # Draw the circuit with a grouped, more structured visualization
# qc.draw('mpl', style={'displaycolor': {'rz': '#0096FF', 'cx': '#0055A4'}})

# # Save the circuit visualization
# plt.savefig("grouped_quantum_circuit.pdf")
# plt.show()

# from qiskit import QuantumCircuit
# from qiskit.quantum_info import SparsePauliOp
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the Hamiltonian as a SparsePauliOp
# hamiltonian = SparsePauliOp.from_list([
#     ('ZIZI', 0.1805827),
#     ('ZIIZ', 0.18884316),
#     ('IZZI', 0.02267623),
#     ('IZIZ', 0.03043532),
#     ('ZIII', -0.73079532),
#     ('IZII', -0.58361906),
#     ('IIZI', -0.98750567),
#     ('IIIZ', -0.9060882)
#     # ('IIII', 0.0) is skipped
# ])

# # Initialize a 4-qubit circuit
# qc = QuantumCircuit(4)

# # Loop over each Pauli term
# for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
#     coeff = coeff.real  # discard imaginary part if 0
#     z_qubits = [i for i, p in enumerate(pauli.to_label()) if p == 'Z']
#     if len(z_qubits) == 2:
#         q0, q1 = z_qubits
#         qc.cx(q0, q1)
#         qc.rz(2 * coeff, q1)  # Typical for ZZ evolution
#         qc.cx(q0, q1)
#     elif len(z_qubits) == 1:
#         qc.rz(2 * coeff, z_qubits[0])
#     # skip if all 'I'

# # Draw and save
# fig = qc.draw('mpl')
# fig.savefig("sparsepauliop_explicit_gates.pdf")
# plt.show()

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import SparsePauliOp, Pauli
import matplotlib.pyplot as plt

# Create custom 2-qubit gate for RXX
def create_rxx_gate(angle):
    qc = QuantumCircuit(2, name=f"RXX\n{angle}")
    rxx_gate = qc.to_gate()
    rxx_gate.label = f"Rₓₓ\n{angle}"
    rxx_gate.name = "RXX"
    return rxx_gate

# Create custom 2-qubit gate for RYY
def create_ryy_gate(angle):
    qc = QuantumCircuit(2, name=f"RYY\n{angle}")
    ryy_gate = qc.to_gate()
    ryy_gate.label = f"Rᵧᵧ\n{angle}"
    ryy_gate.name = "RYY"
    return ryy_gate

# Create local XY mixer Hamiltonian
def create_local_XY_mixer(num_qubits, block_size):
    if num_qubits % block_size != 0:
        raise ValueError("num_qubits must be divisible by block_size.")

    hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
    num_blocks = num_qubits // block_size

    for i in range(num_blocks):
        block_start = i * block_size
        for j in range(0, block_size - 1):
            qubit_1 = block_start + j
            qubit_2 = block_start + j + 1

            xx_term = ['I'] * num_qubits
            yy_term = ['I'] * num_qubits

            xx_term[qubit_1] = 'X'
            xx_term[qubit_2] = 'X'
            yy_term[qubit_1] = 'Y'
            yy_term[qubit_2] = 'Y'

            xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[0.5])
            yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[0.5])

            hamiltonian += xx_op + yy_op

    return hamiltonian

# Params
num_qubits = 4
block_size = 2
theta = 1.0

# Generate mixer
mixer = create_local_XY_mixer(num_qubits, block_size)

# Initialize circuit
qc = QuantumCircuit(num_qubits)

# Add custom RXX and RYY gates for each term
for pauli, coeff in zip(mixer.paulis, mixer.coeffs):
    label = pauli.to_label()
    qubits = [i for i, p in enumerate(label) if p in 'XY']
    if len(qubits) != 2:
        continue
    i, j = qubits
    angle = round(theta * coeff.real, 3)

    if label[i] == 'X' and label[j] == 'X':
        qc.append(create_rxx_gate(angle), [i, j])
    elif label[i] == 'Y' and label[j] == 'Y':
        qc.append(create_ryy_gate(angle), [i, j])

style = {
    'displaycolor': {'RXX': '#0096FF', 'RYY': '#0096FF'},  # Bright blue color for RXX and RYY gates
    'displayindex': False  # Hide qubit indices within gates
}

# Draw the circuit with the specified style and save as a PDF
qc.draw(output='mpl', style=style, filename='XY_mixer.pdf')

# Draw the circuit
plt.show()
