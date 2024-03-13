from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# Initialize the quantum circuit for 4 qubits
qc = QuantumCircuit(4)

# Define the coefficients from the Hamiltonian
coefficients = {
    'ZZII': 6.0,
    'ZIZI': 0.1809819192,
    'ZIIZ': 0.1892471462,
    'IZZI': 0.0227315873,
    'IZIZ': 0.0304955691,
    'IIZZ': 6.0,
    'ZIII': -0.7314345837,
    'IZII': -0.5840745270,
    'IIZI': -0.9879427254,
    'IIIZ': -0.9066044092
}

# Function to add a ZZ operation between two qubits
def add_zz(qc, coeff, q1, q2):
    angle = 2 * coeff  # The angle for RZ is twice the coefficient for ZZ
    qc.cx(q1, q2)
    qc.rz(angle, q2)
    qc.cx(q1, q2)

# Apply the ZZ terms
add_zz(qc, coefficients['ZZII'], 0, 1)
add_zz(qc, coefficients['ZIZI'], 0, 2)
add_zz(qc, coefficients['ZIIZ'], 0, 3)
add_zz(qc, coefficients['IZZI'], 1, 2)
add_zz(qc, coefficients['IZIZ'], 1, 3)
add_zz(qc, coefficients['IIZZ'], 2, 3)


# Apply single Z rotations
qc.rz(2 * -0.7314345837, 0)  # 2 * coefficient for the RZ rotation
qc.rz(2 * -0.5840745270, 1)
qc.rz(2 * -0.9879427254, 2)
qc.rz(2 * -0.9066044092, 3)

# Visualize the circuit
figure = qc.draw('mpl')

figure.savefig("quantum_circuit.pdf")
