# from qiskit import QuantumCircuit
# from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
 
#  # Create empty circuit
# example_circuit = QuantumCircuit(2)
# example_circuit.measure_all()
 
#  # You'll need to specify the credentials when initializing QiskitRuntimeService, if they were not previously saved.
# service = QiskitRuntimeService(channel="ibm_quantum", token="25a4f69c2395dfbc9990a6261b523fe99e820aa498647f92552992afb1bd6b0bbfcada97ec31a81a221c16be85104beb653845e23eeac2fe4c0cb435ec7fc6b4")
# backend = service.backend("ibmq_qasm_simulator")
# job = Sampler(backend).run(example_circuit)
# print(f"job id: {job.job_id()}")
# result = job.result()
# print(result)

# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": 1,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "from qiskit import QuantumCircuit, transpile\n",
#     "from qiskit.quantum_info import SparsePauliOp\n",
#     "from qiskit.primitives import Estimator as Estimator_local\n",
#     "from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options, Sampler\n",
#     "from qiskit_ibm_runtime.options import ResilienceOptions\n",
#     "\n",
#     "import pickle\n",
#     "import numpy as np\n",
#     "import matplotlib.pyplot as plt\n",
#     "\n",
#     "# import deepcopy\n",
#     "import copy\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 2,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# import fake backends\n",
#     "\n",
#     "from qiskit.providers.fake_provider import FakeKolkata\n",
#     "from qiskit_aer.noise import NoiseModel\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 6,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def simulator(circ, resilience_level=1,resilience_options = None):\n",
#     "    \"\"\"Simulate a quantum circuit and estimate its energy using Qiskit Runtime.\n",
#     "\n",
#     "    Args:\n",
#     "        circ (QuantumCircuit): The quantum circuit to be simulated.\n",
#     "        Ham (SparsePauliOp): The Hamiltonian operator of the system.\n",
#     "        resilience_level (int, optional): The level of error mitigation to apply. Defaults to 1.\n",
#     "\n",
#     "    Returns:\n",
#     "        dict: An EstimatorResult containing energy and other information.\n",
#     "    \"\"\"\n",
#     "    # 1. Initialize account\n",
#     "    service = QiskitRuntimeService(channel=\"ibm_quantum\")\n",
#     "\n",
#     "    # 2. Select a backend.\n",
#     "    backend = service.backend(\"ibmq_qasm_simulator\")\n",
#     "\n",
#     "    # 3. make noise model\n",
#     "\n",
#     "    fake_backend = FakeKolkata()\n",
#     "    noise_model = NoiseModel.from_backend(fake_backend)\n",
#     "\n",
#     "\n",
#     "    # 4. Specify options, such as enabling error mitigation\n",
#     "    # optimization_level=3 adds dynamical decoupling\n",
#     "    # resilience_level=1 adds readout error mitigation   \n",
#     "\n",
#     "    options = Options()\n",
#     "\n",
#     "    options.simulator = {\n",
#     "    \"noise_model\": noise_model,\n",
#     "    \"basis_gates\": fake_backend.configuration().basis_gates,\n",
#     "    \"coupling_map\": fake_backend.configuration().coupling_map,\n",
#     "    \"seed_simulator\": 42\n",
#     "    }\n",
#     "    \n",
#     "    options.resilience_level=resilience_level\n",
#     "\n",
#     "    if resilience_level == 0:\n",
#     "        options.optimization_level=0\n",
#     "    else:\n",
#     "        options.optimization_level=3\n",
#     "\n",
#     "    if resilience_options is not None:\n",
#     "        options.resilience.noise_factors = resilience_options.noise_factors\n",
#     "        options.resilience.extrapolator = resilience_options.extrapolator\n",
#     "\n",
#     "    \n",
#     "    with Session(service=service, backend=backend):\n",
#     "        \n",
#     "\n",
#     "        # 5. Create primitive instance\n",
#     "        \n",
#     "        sampler = Sampler(options=options)\n",
#     "        \n",
#     "\n",
#     "        # 6. Submit jobs\n",
#     "        job = sampler.run(circ, shots=1024)\n",
#     "\n",
#     "        # 7. Get results\n",
#     "        print(f\"Job ID: {job.job_id()}\")\n",
#     "        print(f\"Job result: {job.result()}\")\n",
#     "        \n",
#     "        return job\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 4,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     " # Create empty circuit\n",
#     "example_circuit = QuantumCircuit(2)\n",
#     "example_circuit.measure_all()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 7,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "Job ID: cmgjfsimpoeld4n5nl4g\n",
#       "Job result: SamplerResult(quasi_dists=[{0: 0.994241682735211, 1: 0.004829396614733, 2: 0.000928920650056}], metadata=[{'shots': 1024, 'circuit_metadata': {}, 'readout_mitigation_overhead': 1.083475100997166, 'readout_mitigation_time': 0.003052233252674341, 'warning': 'Optimization level clipped from 3 to 1'}])\n"
#      ]
#     },
#     {
#      "data": {
#       "text/plain": [
#        "<RuntimeJob('cmgjfsimpoeld4n5nl4g', 'sampler')>"
#       ]
#      },
#      "execution_count": 7,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "simulator(example_circuit)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "NEO",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.11.5"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 2
# }

from qiskit_aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
from qiskit_ibm_provider import IBMProvider
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeKolkata
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Options, Session, Sampler

IBMProvider.save_account('25a4f69c2395dfbc9990a6261b523fe99e820aa498647f92552992afb1bd6b0bbfcada97ec31a81a221c16be85104beb653845e23eeac2fe4c0cb435ec7fc6b4', overwrite=True)
provider = IBMProvider()
available_backends = provider.backends()
print([backend.name for backend in available_backends])
backend = provider.get_backend('ibmq_qasm_simulator') 
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibmq_qasm_simulator")
noise_model = NoiseModel.from_backend(backend)
simulator = AerSimulator(noise_model = noise_model)
service = QiskitRuntimeService(channel="ibm_quantum")
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
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=mixer_op, initial_point=initial_point)
    result1 = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
print("\n\nThe result of the noisy quantum optimisation using QAOA is: \n")
print('best measurement', result1.best_measurement)