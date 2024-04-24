from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# Initialize the quantum circuit for 4 qubits
qc = QuantumCircuit(2)

# Calculate the combined coefficients
coeff_zz = 0.18098191916942596 - 0.18924714624881744 - 0.02273158729076385
coeff_zi = 0.18098191916942596 + 0.18924714624881744 - 0.02273158729076385 + 0.361205518245697 - 0.5308473706245422
coeff_iz = 0.18098191916942596 - 0.18924714624881744 + 0.02273158729076385 + 0.7842292189598083 - 0.6868616938591005

# Apply the ZZ operation
qc.cx(0, 1)
qc.rz(coeff_zz * 2, 1)  # The angle for the RZ gate is twice the coefficient for ZZ
qc.cx(0, 1)

# Apply the ZI operation (single-qubit Z rotation on qubit 0)
qc.rz(coeff_zi * 2, 0)

# Apply the IZ operation (single-qubit Z rotation on qubit 1)
qc.rz(coeff_iz * 2, 1)


# Visualize the circuit
figure = qc.draw('mpl')

figure.savefig("quantum_circuit.png")


plt.show()