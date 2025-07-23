from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate
import matplotlib.pyplot as plt

# Initialize a quantum circuit with 2 qubits
qc = QuantumCircuit(4)

# Define the mixing angle
theta = 0.5  # Adjust this angle as needed

# Apply the RXX and RYY gates to qubits 0 and 1
qc.rxx(theta, 0, 1)
qc.ryy(theta, 0, 1)

# Define the style to customize gate colors
style = {
    'displaycolor': {'rxx': '#0096FF', 'ryy': '#0096FF'},  # Bright blue color for RXX and RYY gates
    'displayindex': False  # Hide qubit indices within gates
}

# Draw the circuit with the specified style and save as a PDF
qc.draw(output='mpl', style=style, filename='XY_mixer.pdf')

# Display the circuit diagram
plt.show()
