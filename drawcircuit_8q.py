from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# Initialize the quantum circuit for 4 qubits
qc = QuantumCircuit(8)

# Define the coefficients from the Hamiltonian
coefficients = {
    (0, 1): 6.0,  # ZZIIIIII
    (0, 2): 0.7874419689,  # ZIZIIIII
    (0, 3): 0.0684041679,  # ZIIZIIII
    (1, 2): 0.2517849803,  # IZZIIIII
    (1, 3): -0.0734889582,  # IZIZIIII
    (2, 3): 6.0,  # IIZZIIII
    (2, 4): 0.1819513440, #IIZIZIII
    (2, 5): 0.1084431708, #IIZIIZII
    (3, 4): 0.0962224975, #IIIZZIII
    (3, 5): 0.0227499232, # IIIZIZII
    (4, 5): 6.0, # IIIIZZII
    (4, 6): -0.2104287297, #IIIIZIZI
    (4, 7): -0.2377609015,#IIIIZIIZ
    (5, 6):  0.0405503064, # IIIIIZZI
    (5, 7): 0.0142428726, # 
    (6, 7): 6.0 # IIIIIIZZ
}

for (q1, q2), coeff in coefficients.items():
    qc.cx(q1, q2)
    qc.rz(2 * coeff, q2)  # The rotation angle might be 2 * coeff, check your Hamiltonian encoding
    qc.cx(q1, q2)


single_z_coefficients = [
    -1.6783797443,  # ZIIIIIII
    -0.8278056309,  # IZIIIIII
    -2.1113303006,  # IIZIIIII
    -0.4333506450,  # IIIZIIII
    -0.4527192339,  # IIIIZIII
    -0.5309761092,  # IIIIIZII
    -0.6598197818,  # IIIIIIZI
    -0.7743371576,  # IIIIIIIZ
]

# Apply single Z rotations
for qubit, coeff in enumerate(single_z_coefficients):
    qc.rz(2 * coeff, qubit)  # The rotation angle might be 2 * coeff, check your Hamiltonian encoding

# Visualize the circuit
figure = qc.draw('mpl')

figure.savefig("4res_2rot.png")

plt.show()