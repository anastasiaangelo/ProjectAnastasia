# %%
# imports

import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os

from supporting_functions import *

def noisy_simulation(num_rot, num_res, alpha, shots, p):

    # %%
    # parameters

    # num_rot = 3
    # num_res = 4
    num_qubits = num_rot * num_res
    # alpha = 1
    # shots = 5000
    # p = 1

    # %%
    H = get_hamiltonian(num_rot, num_res)
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)
    XY_mixer = get_XY_mixer(num_qubits, num_rot)

    # %%
    from qiskit_algorithms.minimum_eigensolvers import QAOA
    from qiskit.quantum_info.operators import Pauli, SparsePauliOp
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Sampler
    from qiskit import QuantumCircuit






    # %%
    initial_point = np.ones(2 * p)
    initial_bitstring = generate_initial_bitstring(num_qubits, num_rot)
    state_vector = np.zeros(2**num_qubits)
    indexx = int(initial_bitstring, 2)
    state_vector[indexx] = 1
    qc = QuantumCircuit(num_qubits)
    qc.initialize(state_vector, range(num_qubits))


    # %%
    from qiskit_aer import Aer
    from qiskit_ibm_provider import IBMProvider
    from qiskit_aer.noise import NoiseModel
    from qiskit.primitives import BackendSampler
    from qiskit.transpiler import PassManager

    # %%
    simulator = Aer.get_backend('qasm_simulator')
    provider = IBMProvider()

    device_backend = provider.get_backend('ibm_torino')
    noise_model = NoiseModel.from_backend(device_backend)

    # %%
    options= {
        "noise_model": noise_model,
        "basis_gates": simulator.configuration().basis_gates,
        "coupling_map": simulator.configuration().coupling_map,
        "seed_simulator": 42,
        "shots": shots,
        "optimization_level": 3,
        "resilience_level": 3
    }

    # %%
    def callback(quasi_dists, parameters, energy):
        intermediate_data.append(
            quasi_dists
        )

    # %%
    intermediate_data = []
    noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())

    # %%
    start_time1 = time.time()
    qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point,callback=callback, aggregation=alpha)
    result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    # %%
    # %% Post Selection



    # %%
    intermediate_data_dicts = []

    for item in intermediate_data:
        for dict_item in item:
            intermediate_data_dicts.append(dict_item)

    # %%

    probability = []
    total_arr = []
    cumulative_probability_dict = {}
    cumulative_total_dict = {}

    for i, dict in enumerate(intermediate_data_dicts):
        #print(f"\n\nIteration {i}")
        #print(f"Dictionary: {dict}")
        hits = 0.0
        total = 0.0
        for key in dict:
            bitstring = int_to_bitstring(key, num_qubits)
            #print(f"\nBitstring: {bitstring}")
            hamming = check_hamming(bitstring, num_rot)
        #  print(f"Hamming condition: {hamming}")
            if check_hamming(bitstring, num_rot):
                hits += dict[key]
                total += dict[key]
                #print(f"Bitstring: {bitstring} has a value of {dict[key]}")
                if bitstring in cumulative_probability_dict:
                    cumulative_probability_dict[bitstring] += dict[key]
                else:
                    cumulative_probability_dict[bitstring] = dict[key]
            else:
                total += dict[key]
            if bitstring in cumulative_total_dict:
                cumulative_total_dict[bitstring] += dict[key]
            else:
                cumulative_total_dict[bitstring] = dict[key]
                #print(f"Bitstring: {bitstring} does not satisfy the Hamming condition.")
                #pass
        
        probability.append(hits)
        total_arr.append(total)

    # %%
    # sum the values of the cumulative_probability_dict and cumulative_total_dict

    sum_total = sum(cumulative_total_dict.values())
    sum_probability = sum(cumulative_probability_dict.values())

    # print(f"Total probability: {sum_probability}, Total: {sum_total}")

    norm = sum_total
    fraction = sum_probability / sum_total




    # %%
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt




    # %%
    eigenstate_distribution = result1.eigenstate
    best_measurement = result1.best_measurement
    final_bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

    all_bitstrings = {}
    all_unrestricted_bitstrings = {}

    for state, prob in final_bitstrings.items():
        bitstring = int_to_bitstring(state, num_qubits)
        energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
        if bitstring not in all_unrestricted_bitstrings:
            all_unrestricted_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
        all_unrestricted_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
        all_unrestricted_bitstrings[bitstring]['energy'] = (all_unrestricted_bitstrings[bitstring]['energy'] * all_unrestricted_bitstrings[bitstring]['count'] + energy) / (all_unrestricted_bitstrings[bitstring]['count'] + 1)
        all_unrestricted_bitstrings[bitstring]['count'] += 1

        if check_hamming(bitstring, num_rot):
            if bitstring not in all_bitstrings:
                all_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
            all_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
            all_bitstrings[bitstring]['energy'] = (all_bitstrings[bitstring]['energy'] * all_bitstrings[bitstring]['count'] + energy) / (all_bitstrings[bitstring]['count'] + 1)
            all_bitstrings[bitstring]['count'] += 1

            ## here, count is not related to the number of counts of the optimiser,
            ## it keeps track of number of times the bitstring has been seen in 
            ## different iterations of the optimiser. This is used to calculate the
            ## average energy of the bitstring across iterations. Ideally this should
            ## be weighted by the probability of the bitstring in each iteration.
            ## For the moment the energy is calculated by the statevector simulator,
            ## so it should be fine. ##TODO : Adapt this for noisy simulations.


    for data in intermediate_data_dicts:
        
        for int_bitstring in data:
            probability = data[int_bitstring]
            intermediate_bitstring = int_to_bitstring(int_bitstring, num_qubits)
            energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)
            if bitstring not in all_unrestricted_bitstrings:
                all_unrestricted_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
            all_unrestricted_bitstrings[bitstring]['probability'] += probability  # Aggregate probabilities
            all_unrestricted_bitstrings[bitstring]['energy'] = (all_unrestricted_bitstrings[bitstring]['energy'] * all_unrestricted_bitstrings[bitstring]['count'] + energy) / (all_unrestricted_bitstrings[bitstring]['count'] + 1)
            all_unrestricted_bitstrings[bitstring]['count'] += 1

            if check_hamming(intermediate_bitstring, num_rot):
                if intermediate_bitstring not in all_bitstrings:
                    all_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
                all_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
                
                all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * all_bitstrings[intermediate_bitstring]['count'] + energy) / (all_bitstrings[intermediate_bitstring]['count'] + 1)
                all_bitstrings[intermediate_bitstring]['count'] += 1

    # %%
    sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])
    sorted_unrestricted_bitstrings = sorted(all_unrestricted_bitstrings.items(), key=lambda x: x[1]['energy'])

    # %%
    # Store information
    probabilities = []
    sorted_bitstrings_arr = []


    probabilities = [data['probability'] for bitstring, data in sorted_bitstrings]
            

    sorted_bitstrings_arr = [bitstring for bitstring, data in sorted_bitstrings]

    probabilities = np.array(probabilities) / norm

    # %%
    result = {
        'params' : (num_res, num_rot, alpha, shots, p),
        'bitstrings': sorted_bitstrings_arr,
        'probabilities': probabilities,
        'fraction': fraction,
        'norm': norm,
        'energy': sorted_bitstrings[0][1]['energy'],
        'elapsed_time': elapsed_time1,
        'intermediate_data': intermediate_data,
        'cumulative_probability_dict': cumulative_probability_dict,
        'cumulative_total_dict': cumulative_total_dict,
        'all_bitstrings': all_bitstrings,
        'all_unrestricted_bitstrings': all_unrestricted_bitstrings,
        'sorted_bitstrings': sorted_bitstrings,
        'sorted_unrestricted_bitstrings': sorted_unrestricted_bitstrings
    }

    import json

    # write json encoder


        
    # write json file

    with open(f"RESULTS/XY-QAOA/JSON/{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p.json", 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)

    # with open(f"RESULTS/XY-QAOA/JSON/{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p.json", 'w') as f:
    #     json.dump(result, f)



    # %%
    # # Plotting the ratios
    # fig, ax = plt.subplots()

    # x3 = range(len(sorted_bitstrings_arr))

    # # ax.plot(x, y_XY, label="XY mixer 2rot", marker='x')
    # ax.plot(x3, probabilities, label=f"XY mixer 3 rot alpha : {alpha}", marker='1')

    # ax.set_xlabel("Bitstrings")
    # ax.set_ylabel("Probability")
    # ax.set_title("Probability distributions of good bitstrings by Sorted Bitstrings")
    # ax.legend()

    # ax.set_xticks(x3)

    # plt.tight_layout()
    # # savefig with params in file name

    # plt.savefig(f"./RESULTS/XY-QAOA/CVaR/prob_distributions_alpha_{alpha}.pdf")
    # plt.show()

    # %%


    # %%


    # %%



