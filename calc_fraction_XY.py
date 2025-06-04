
# %%
import pyrosetta
from qiskit import QuantumCircuit
pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *

import numpy as np
import pandas as pd

from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack.task import *

# Initiate structure, scorefunction, change PDB files
num_ress = [6]
num_rots = range(2,3)

for num_res in num_ress:
    for num_rot in num_rots:
        file_path = f"RESULTS/{num_rot}rot-XY-QAOA/{num_res}res-{num_rot}rot.csv"

        pose = pyrosetta.pose_from_pdb(f"input_files/{num_res}residue.pdb")


        residue_count = pose.total_residue()
        sfxn = get_score_function(True)
        print(pose.sequence())
        print(residue_count)


        relax_protocol = pyrosetta.rosetta.protocols.relax.FastRelax()
        relax_protocol.set_scorefxn(sfxn)
        relax_protocol.apply(pose)

        # Define task, interaction graph and rotamer sets (model_protein_csv.py)
        task_pack = TaskFactory.create_packer_task(pose) 

        rotsets = RotamerSets()
        pose.update_residue_neighbors()
        sfxn.setup_for_packing(pose, task_pack.designing_residues(), task_pack.designing_residues())
        packer_neighbor_graph = pyrosetta.rosetta.core.pack.create_packer_graph(pose, sfxn, task_pack)
        rotsets.set_task(task_pack)
        rotsets.build_rotamers(pose, sfxn, packer_neighbor_graph)
        rotsets.prepare_sets_for_packing(pose, sfxn) 
        ig = InteractionGraphFactory.create_interaction_graph(task_pack, rotsets, pose, sfxn, packer_neighbor_graph)
        print("built", rotsets.nrotamers(), "rotamers at", rotsets.nmoltenres(), "positions.")
        rotsets.compute_energies(pose, sfxn, packer_neighbor_graph, ig, 1)

        # Output structure to be visualised in pymol
        pose.dump_pdb("output_repacked.pdb")

        # Define dimension for matrix
        max_rotamers = 0
        for residue_number in range(1, residue_count+1):
            n_rots = rotsets.nrotamers_for_moltenres(residue_number)
            print(f"Residue {residue_number} has {n_rots} rotamers.")
            if n_rots > max_rotamers:
                max_rotamers = n_rots

        print("Maximum number of rotamers:", max_rotamers)


        E = np.zeros((max_rotamers, max_rotamers))
        Hamiltonian = np.zeros((max_rotamers, max_rotamers))

        E1 = np.zeros((max_rotamers, max_rotamers))
        Hamiltonian1 = np.zeros((max_rotamers, max_rotamers))

        output_file = "energy_files/two_body_terms.csv"
        output_file1 = "energy_files/one_body_terms.csv"
        data_list = []
        data_list1 = []
        df = pd.DataFrame(columns=['res i', 'res j', 'rot A_i', 'rot B_j', 'E_ij'])
        df1 = pd.DataFrame(columns=['res i', 'rot A_i', 'E_ii'])


        # # Visualisation of structure after repacking with rotamers
        # pmm = PyMOLMover()
        # clone_pose = Pose()
        # clone_pose.assign(pose)
        # pmm.apply(clone_pose)
        # pmm.send_hbonds(clone_pose)
        # pmm.keep_history(True)
        # pmm.apply(clone_pose)

        # to limit to n rotamers per residue, change based on how many rotamers desired
        # num_rot = 2

        # Loop to find Hamiltonian values Jij - interaction of rotamers on NN residues
        for residue_number in range(1, residue_count):
            rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
            if rotamer_set_i == None: # skip if no rotamers for the residue
                continue

            residue_number2 = residue_number + 1
            residue2 = pose.residue(residue_number2)
            rotamer_set_j = rotsets.rotamer_set_for_residue(residue_number2)
            if rotamer_set_j == None:
                continue

            molten_res_i = rotsets.resid_2_moltenres(residue_number)
            molten_res_j = rotsets.resid_2_moltenres(residue_number2)

            edge_exists = ig.find_edge(molten_res_i, molten_res_j)
                
            if not edge_exists:
                    continue

            for rot_i in range(1, rotamer_set_i.num_rotamers() + 1):
                for rot_j in range(1, rotamer_set_j.num_rotamers() + 1):
                    E[rot_i-1, rot_j-1] = ig.get_two_body_energy_for_edge(molten_res_i, molten_res_j, rot_i, rot_j)
                    Hamiltonian[rot_i-1, rot_j-1] = E[rot_i-1, rot_j-1]

            for rot_i in range(10, num_rot + 10):       #, rotamer_set_i.num_rotamers() + 1):
                for rot_j in range(10, num_rot + 10):       #, rotamer_set_j.num_rotamers() + 1):
                    # print(f"Interaction energy between rotamers of residue {residue_number} rotamer {rot_i} and residue {residue_number2} rotamer {rot_j} :", Hamiltonian[rot_i-1, rot_j-1])
                    data = {'res i': residue_number, 'res j': residue_number2, 'rot A_i': rot_i, 'rot B_j': rot_j, 'E_ij': Hamiltonian[rot_i-1, rot_j-1]}
                    data_list.append(data)

        # Save the two-body energies to a csv file
        df = pd.DataFrame(data_list)
        df.to_csv('energy_files/two_body_terms.csv', index=False)

        # to choose the two rotamers with the largest energy in absolute value
        # df.assign(abs_E=df['E_ij'].abs()).nlargest(2, 'abs_E').drop(columns=['abs_E']).to_csv('two_body_terms.csv', index=False)

        # Loop to find Hamiltonian values Jii
        for residue_number in range(1, residue_count + 1):
            residue1 = pose.residue(residue_number)
            rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
            if rotamer_set_i == None: 
                continue

            molten_res_i = rotsets.resid_2_moltenres(residue_number)

            for rot_i in range(10, num_rot +10):        #, rotamer_set_i.num_rotamers() + 1):
                E1[rot_i-1, rot_i-1] = ig.get_one_body_energy_for_node_state(molten_res_i, rot_i)
                Hamiltonian1[rot_i-1, rot_i-1] = E1[rot_i-1, rot_i-1]
                # print(f"Interaction score values of {residue1.name3()} rotamer {rot_i} with itself {Hamiltonian[rot_i-1,rot_i-1]}")
                data1 = {'res i': residue_number, 'rot A_i': rot_i, 'E_ii': Hamiltonian1[rot_i-1, rot_i-1]}
                data_list1.append(data1)
            


        # Save the one-body energies to a csv file
        df1 = pd.DataFrame(data_list1)
        df1.to_csv('energy_files/one_body_terms.csv', index=False)
        # to choose the two rotamers with the largest energy in absolute value
        # df1.assign(abs_Ei=df1['E_ii'].abs()).nlargest(2, 'abs_Ei').drop(columns=['abs_Ei']).to_csv('one_body_terms.csv', index=False)

        import numpy as np
        import pandas as pd
        import time
        from copy import deepcopy
        import os

        ########################### Configure the hamiltonian from the values calculated classically with pyrosetta ############################
        df1 = pd.read_csv("energy_files/one_body_terms.csv")
        q = df1['E_ii'].values
        num = len(q)
        N = int(num/num_rot)
        num_qubits = num

        print('Qii values: \n', q)

        df2 = pd.read_csv("energy_files/two_body_terms.csv")
        value = df2['E_ij'].values
        Q = np.zeros((num,num))
        n = 0

        for j in range(0, num-num_rot, num_rot):
            for i in range(j, j+num_rot):
                for offset in range(num_rot):
                    Q[i][j+num_rot+offset] = deepcopy(value[n])
                    Q[j+num_rot+offset][i] = deepcopy(value[n])
                    n += 1

        print('\nQij values: \n', Q)

        H = np.zeros((num,num))

        for i in range(num):
            for j in range(num):
                if i != j:
                    H[i][j] = np.multiply(0.25, Q[i][j])

        for i in range(num):
            H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

        print('\nH: \n', H)

        k = 0
        for i in range(num_qubits):
            k += 0.5 * q[i]

        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    k += 0.5 * 0.25 * Q[i][j]

    
        # %% ############################################ Quantum hamiltonian ########################################################################
        from qiskit_algorithms.minimum_eigensolvers import QAOA
        from qiskit.quantum_info.operators import Pauli, SparsePauliOp
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.primitives import Sampler

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

        def format_sparsepauliop(op):
            terms = []
            labels = [pauli.to_label() for pauli in op.paulis]
            coeffs = op.coeffs
            for label, coeff in zip(labels, coeffs):
                terms.append(f"{coeff:.10f} * {label}")
            return '\n'.join(terms)

        print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))


        def get_XY_mixer(num_qubits, num_rot, transverse_field=1):
            if num_rot < 2:
                raise ValueError("num_rot must be at least 2.")

            hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])

            for i in range(0, num_qubits - num_rot + 1, num_rot):  
                pauli_terms = {'X': [], 'Y': []}
                
                # Generate all pairwise (X, X) and (Y, Y) terms in each `num_rot` group
                for j in range(num_rot):
                    for k in range(j + 1, num_rot):
                        xx_term = ['I'] * num_qubits
                        yy_term = ['I'] * num_qubits
                        xx_term[i + j] = 'X'
                        xx_term[i + k] = 'X'
                        yy_term[i + j] = 'Y'
                        yy_term[i + k] = 'Y'

                        xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
                        yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])

                        hamiltonian += xx_op + yy_op

            hamiltonian *= transverse_field
            return -hamiltonian if num_rot == 2 else hamiltonian

        XY_mixer = get_XY_mixer(num_qubits, num_rot)

        p = 1  # Number of QAOA layers
        initial_point = np.ones(2 * p)
        intermediate_data = []
        def callback(quasi_dists, parameters, energy):
            intermediate_data.append({
                'quasi_distributions': quasi_dists,
                'parameters': parameters,
                'energy': energy
            })

        options= {
        "seed_simulator": 42,
        "shots": 1000,
        "max_parallel_threads" : 0,
        "max_parallel_experiments" : 0,
        "max_parallel_shots" : 1,
        "statevector_parallel_threshold" : 16
    }
        

        def generate_initial_bitstring(num_qubits, num_rot):
            if num_rot < 2:
                raise ValueError("num_rot must be at least 2.")

            pattern = ['0'] * num_rot  
            pattern[0] = '1'  # Ensure at least one '1' per group
            bitstring = ''.join(pattern * ((num_qubits // num_rot) + 1))[:num_qubits]

            return bitstring

        sampler = Sampler(options=options)
        initial_point = np.ones(2 * p)
        initial_bitstring = generate_initial_bitstring(num_qubits, num_rot)
        state_vector = np.zeros(2**num_qubits)
        indexx = int(initial_bitstring, 2)
        state_vector[indexx] = 1
        qc = QuantumCircuit(num_qubits)
        qc.initialize(state_vector, range(num_qubits))


        # %% Local simulation, too slow when big sizes
        start_time = time.time()
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, mixer=XY_mixer, initial_point=initial_point, initial_state=qc, callback=callback)
        result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
        end_time = time.time()

        print("\n\nThe result of the quantum optimisation using QAOA is: \n")
        print('best measurement', result.best_measurement)
        elapsed_time = end_time - start_time
        print(f"Local Simulation run time: {elapsed_time} seconds")
        print('\n\n')

        # %%
        from qiskit_aer.primitives import Estimator
        from qiskit_aer import Aer
        from qiskit import QuantumCircuit, transpile

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


        eigenstate_distribution = result.eigenstate
        best_measurement = result.best_measurement
        final_bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

        all_bitstrings = {}
        for state, prob in final_bitstrings.items():
            bitstring = int_to_bitstring(state, num_qubits)
            if check_hamming(bitstring, num_rot):
                if bitstring not in all_bitstrings:
                    all_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
                all_bitstrings[bitstring]['probability'] += prob
                energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
                all_bitstrings[bitstring]['energy'] = (all_bitstrings[bitstring]['energy'] * all_bitstrings[bitstring]['count'] + energy) / (all_bitstrings[bitstring]['count'] + 1)
                all_bitstrings[bitstring]['count'] += 1


        for data in intermediate_data:
            print(f"Quasi Distribution: {data['quasi_distributions']}, Parameters: {data['parameters']}, Energy: {data['energy']}")
            for distribution in data['quasi_distributions']:
                for int_bitstring, probability in distribution.items():
                    intermediate_bitstring = int_to_bitstring(int_bitstring, num_qubits)
                    if check_hamming(intermediate_bitstring, num_rot):
                        if intermediate_bitstring not in all_bitstrings:
                            all_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
                        all_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
                        energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)
                        all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * all_bitstrings[intermediate_bitstring]['count'] + energy) / (all_bitstrings[intermediate_bitstring]['count'] + 1)
                        all_bitstrings[intermediate_bitstring]['count'] += 1


                # Sorting bitstrings by energy
        sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])

        # Calculating the fraction of bitstrings that satisfy the Hamming condition
        total_bitstrings = sum(probability * options['shots'] for data in intermediate_data
                            for distribution in data['quasi_distributions']
                            for int_bitstring, probability in distribution.items()) + \
                            sum(probability * options['shots'] for state, probability in final_bitstrings.items())

        hamming_satisfying_bitstrings = sum(bitstring_data['probability'] * options['shots'] for bitstring_data in all_bitstrings.values())

        fraction_satisfying_hamming = hamming_satisfying_bitstrings / total_bitstrings
        print(f"Fraction of bitstrings that satisfy the Hamming constraint: {fraction_satisfying_hamming}")

        # If you need to track the "best" bitstring (lowest energy)
        best_bitstring, best_bitstring_data = sorted_bitstrings[0]
        print(f"Best bitstring: {best_bitstring}, Energy: {best_bitstring_data['energy']}")

        # Example of tracking the ground state
        found = False
        for bitstring, data in sorted_bitstrings:
            if bitstring == best_measurement['bitstring']:
                print('Best measurement bitstring respects Hamming conditions.\n')
                print('Ground state energy: ', data['energy'] + k)
                # Save the data for this bitstring
                data = {
                    "Experiment": ["Aer Simulation Local Penalty QAOA"],
                    "Ground State Energy": [np.real(result.best_measurement['value'] + k)],
                    "Best Measurement": [result.best_measurement],
                    "Number of qubits": [num_qubits],
                    "shots": [options['shots']],
                    "Fraction": [fraction_satisfying_hamming]
                }
                found = True
                break

        # If no satisfying ground state found, use the sorted lowest-energy bitstring
        if not found:
            print('Best measurement bitstring does not respect Hamming conditions, taking the sorted bitstring corresponding to the smallest energy.\n')
            post_selected_bitstring, post_selected_energy = sorted_bitstrings[0]
            data = {
                "Experiment": ["Aer Simulation Local Penalty QAOA, post-selected"],
                "Ground State Energy": [post_selected_energy['energy'] + k],
                "Best Measurement": [post_selected_bitstring],
                "Number of qubits": [num_qubits],
                "Fraction": [fraction_satisfying_hamming]
            }

        # Save to CSV
        df = pd.DataFrame(data)
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', index=False, header=False)
