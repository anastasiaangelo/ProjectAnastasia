### Take input pdb, score, repack and extract one and two body energies
# Script for generating the test sturcture with its rotamers 
#  lines 23 and 87 to vary
import pyrosetta
pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *

import csv
import os
from copy import deepcopy
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta import PyMOLMover

# Initiate structure, scorefunction, change PDB files
num_res = range(8, 16)
num_rot = range(2, 3)

for res in num_res:
    for rot in num_rot:
        pose = pyrosetta.pose_from_pdb(f"input_files/{res}residue.pdb")


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

            for rot_i in range(10, rot + 10):       #, rotamer_set_i.num_rotamers() + 1):
                for rot_j in range(10, rot + 10):       #, rotamer_set_j.num_rotamers() + 1):
                    # print(f"Interaction energy between rotamers of residue {residue_number} rotamer {rot_i} and residue {residue_number2} rotamer {rot_j} :", Hamiltonian[rot_i-1, rot_j-1])
                    data = {'res i': residue_number, 'res j': residue_number2, 'rot A_i': rot_i, 'rot B_j': rot_j, 'E_ij': Hamiltonian[rot_i-1, rot_j-1]}
                    data_list.append(data)

        # Save the two-body energies to a csv file
        df = pd.DataFrame(data_list)
        df.to_csv('energy_files/two_body_terms.csv', index=False)

        # Loop to find Hamiltonian values Jii
        for residue_number in range(1, residue_count + 1):
            residue1 = pose.residue(residue_number)
            rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
            if rotamer_set_i == None: 
                continue

            molten_res_i = rotsets.resid_2_moltenres(residue_number)

            for rot_i in range(10, rot +10):        #, rotamer_set_i.num_rotamers() + 1):
                E1[rot_i-1, rot_i-1] = ig.get_one_body_energy_for_node_state(molten_res_i, rot_i)
                Hamiltonian1[rot_i-1, rot_i-1] = E1[rot_i-1, rot_i-1]
                # print(f"Interaction score values of {residue1.name3()} rotamer {rot_i} with itself {Hamiltonian[rot_i-1,rot_i-1]}")
                data1 = {'res i': residue_number, 'rot A_i': rot_i, 'E_ii': Hamiltonian1[rot_i-1, rot_i-1]}
                data_list1.append(data1)
            
        # Save the one-body energies to a csv file
        df1 = pd.DataFrame(data_list1)
        df1.to_csv('energy_files/one_body_terms.csv', index=False)
 
#         ########################### Configure the hamiltonian from the values calculated classically with pyrosetta ############################
        df1 = pd.read_csv("energy_files/one_body_terms.csv")
        q = df1['E_ii'].values
        num = len(q)
        N = int(num/rot)
        num_qubits = num

        print('Qii values: \n', q)

        df = pd.read_csv("energy_files/two_body_terms.csv")
        value = df['E_ij'].values
        Q = np.zeros((num,num))
        n = 0

        for j in range(0, num-rot, rot):
            for i in range(j, j+rot):
                for offset in range(rot):
                    Q[i][j+rot+offset] = deepcopy(value[n])
                    Q[j+rot+offset][i] = deepcopy(value[n])
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

        from qiskit.quantum_info.operators import Pauli, SparsePauliOp

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

        q_hamiltonian_XY = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if H[i][j] != 0:
                    pauli = generate_pauli_zij(num_qubits, i, j)
                    op = SparsePauliOp(pauli, coeffs=[H[i][j]])
                    q_hamiltonian_XY += op

        for i in range(num_qubits):
            pauli = generate_pauli_zij(num_qubits, i, i)
            Z_i = SparsePauliOp(pauli, coeffs=[H[i][i]])
            q_hamiltonian_XY += Z_i

        def format_sparsepauliop(op):
            terms = []
            labels = [pauli.to_label() for pauli in op.paulis]
            coeffs = op.coeffs
            for label, coeff in zip(labels, coeffs):
                terms.append(f"{coeff:.10f} * {label}")
            return '\n'.join(terms)

        print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian_XY))
        import networkx as nx
        def create_xy_hamiltonian(num_qubits):
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

        XY_mixer = create_xy_hamiltonian(num_qubits)

# ################################## no penalty depth ##############################
        # file_name = f"RESULTS/qH_depths/total_cnots_{rot}rots_nopenalty.csv"
        file_name_XY = f"RESULTS/qH_depths/total_depth_{rot}rots_nopenalty.csv"

        def pauli_shares_qubits(pauli1, pauli2):
            """
            Returns True if the two Pauli strings share any qubit with non-identity ops.
            """
            for p1, p2 in zip(pauli1, pauli2):
                if p1 != 'I' and p2 != 'I':
                    return True
            return False


        def compute_commuting_layers(hamiltonian):
            """
            Uses graph coloring to compute the exact number of layers for ZZ terms based on qubit overlaps.
            Separates ZZ interaction layers and single-qubit Z layers for more accurate depth estimation.

            Parameters:
            - hamiltonian: SparsePauliOp representing the Hamiltonian.

            Returns:
            - depth_HC: The total depth required for compiled H_C.
            - num_ZZ_layers: The number of parallelizable ZZ layers.
            - num_single_Z_layers: The number of single-qubit Z layers.
            - layer_assignments: Dictionary mapping each term to its assigned layer.
            """
            pauli_labels = [pauli.to_label() for pauli in hamiltonian.paulis]

            # Separate ZZ terms and single-qubit Z terms
            two_qubit_terms = [label for label in pauli_labels if sum(p != 'I' for p in label) == 2]
            single_z_terms = [label for label in pauli_labels if label.count('Z') == 1]  # Single-qubit rotations

            # Step 1: Create Conflict Graph (Nodes = ZZ terms, Edges = Shared Qubits)
            G = nx.Graph()
            for term in two_qubit_terms:
                G.add_node(term)  # Each ZZ term is a node

            for i, term1 in enumerate(two_qubit_terms):
                for j, term2 in enumerate(two_qubit_terms):
                    if i < j and pauli_shares_qubits(term1, term2):
                        G.add_edge(term1, term2)  # Conflict edge (they share a qubit)

            # Step 2: Solve Graph Coloring Problem (to find minimum layers for ZZ terms)
            coloring = nx.coloring.greedy_color(G, strategy="largest_first")

            # Step 3: Assign ZZ Layers
            num_ZZ_layers = max(coloring.values()) + 1 if coloring else 0
            layer_assignments = {term: layer for term, layer in coloring.items()}

            # Step 4: Assign Single-Qubit Z Layers (all single-qubit Z can be parallel in one layer)
            num_single_Z_layers = 1 if single_z_terms else 0
            for term in single_z_terms:
                layer_assignments[term] = num_ZZ_layers  # Put single-qubit Z terms in their own separate layer

            # Compute Corrected Depth: (ZZ layers * 3) + (Single-Z layers * 1)
            # depth_HC = (num_ZZ_layers * 3) + (num_single_Z_layers * 1)
            depth_HC = (num_ZZ_layers * 3) 

            return depth_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments

        def compute_qaoa_depth(cost_hamiltonian, mixer_hamiltonian=None, mixer_type="X"):
            """
            Computes the estimated depth of a full QAOA layer.

            Parameters:
            - cost_hamiltonian: SparsePauliOp for H_C.
            - mixer_hamiltonian: SparsePauliOp for H_B (if XY).
            - mixer_type: "X" or "XY".

            Returns:
            - D_HC: depth of the cost Hamiltonian circuit (compiled).
            - D_QAOA_layer: total depth of one QAOA layer.
            - details: dictionary with breakdown of depths and layers.
            """

            # Count CNOTs from two-qubit ZZ terms (each one contributes 2 CNOTs)
            pauli_labels = [pauli.to_label() for pauli in cost_hamiltonian.paulis]
            two_qubit_terms = [label for label in pauli_labels if label.count('Z') == 2 and label.count('I') == len(label) - 2]
            num_cnot_cost = 2 * len(two_qubit_terms)

            # CNOTs from XY mixer if used
            num_cnot_mixer = 0
            if mixer_type == "XY" and mixer_hamiltonian is not None:
                mixer_labels = [pauli.to_label() for pauli in mixer_hamiltonian.paulis]
                two_qubit_mixer_terms = [label for label in mixer_labels if sum(p != 'I' for p in label) == 2]
                num_cnot_mixer = 2 * len(two_qubit_mixer_terms)  # assuming similar CX-RZ-CX or equivalent

            # Total CNOTs per QAOA layer
            total_cnot = num_cnot_cost + num_cnot_mixer

            # Cost Hamiltonian depth
            D_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(cost_hamiltonian)

            if mixer_type == "X":
                D_HB = 1
                num_mixer_layers = 1
            elif mixer_type == "XY":
                if mixer_hamiltonian is None:
                    raise ValueError("XY mixer selected but no mixer Hamiltonian provided.")
                D_HB, num_XY_layers, _, _ = compute_commuting_layers(mixer_hamiltonian)
                num_mixer_layers = num_XY_layers
            else:
                raise ValueError("Unknown mixer type.")

            D_QAOA_layer = num_ZZ_layers + num_mixer_layers

            details = {
                "D_HC": D_HC,
                "D_HB": D_HB,
                "D_QAOA_layer": D_QAOA_layer,
                "num_ZZ_layers": num_ZZ_layers,
                "num_single_Z_layers": num_single_Z_layers,
                "num_mixer_layers": num_mixer_layers,
                "mixer_type": mixer_type,
                "CNOTs": total_cnot
            }

            return D_HC, D_QAOA_layer, total_cnot, layer_assignments, details

        from qiskit import QuantumCircuit

        def build_qaoa_circuit(layer_assignments, cost_hamiltonian, mixer_type="X", reps=1, gamma=1.0, beta=0.8):
            num_qubits = cost_hamiltonian.num_qubits
            qc = QuantumCircuit(num_qubits)

            # Initial state: put all qubits into superposition
            qc.h(range(num_qubits))

            # Extract layers: group ZZ/Z terms
            max_layer = max(layer_assignments.values())
            layers = {i: [] for i in range(max_layer + 1)}
            for label, layer in layer_assignments.items():
                layers[layer].append(label)

            for rep in range(reps):
                ### Cost Hamiltonian
                for layer_idx in sorted(layers.keys()):
                    for pauli_label in layers[layer_idx]:
                        coeff_idx = list(cost_hamiltonian.paulis.to_labels()).index(pauli_label)
                        coeff = float(cost_hamiltonian.coeffs[coeff_idx].real)  # assume real coeffs

                        if pauli_label.count('Z') == 1:
                            # Single-qubit Z: Z_i → RZ(2γ*coeff)
                            i = pauli_label.index('Z')
                            qc.rz(2 * gamma * coeff, i)

                        elif pauli_label.count('Z') == 2:
                            i, j = [k for k, op in enumerate(pauli_label) if op == 'Z']
                            qc.cx(i, j)
                            qc.rz(2 * gamma * coeff, j)
                            qc.cx(i, j)

                        else:
                            raise ValueError(f"Unsupported Pauli term: {pauli_label}")

                ### Mixer Hamiltonian
                if mixer_type == "X":
                    for i in range(num_qubits):
                        qc.rx(2 * beta, i)

                elif mixer_type == "XY":
                    for i in range(0, num_qubits - 1, 2):
                        # Implement XY interaction: X_i X_{i+1} + Y_i Y_{i+1}
                        qc.rxx(2 * beta, i, i + 1)
                        qc.ryy(2 * beta, i, i + 1)

                else:
                    raise ValueError("Unknown mixer type")

            return qc


        D_HC, D_QAOA_layer, total_cnot, layer_assignments, details = compute_qaoa_depth(q_hamiltonian_XY, XY_mixer, "XY")
        # D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(q_hamiltonian)

        qc = build_qaoa_circuit(layer_assignments, q_hamiltonian_XY, mixer_type="XY", reps=1)
        depth_qc = qc.depth()
        print(f"\n Quantum Circuit Depth: {depth_qc}")

        layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
        layer_df = layer_df.sort_values(by="Assigned Layer")

        size = num_qubits
        depth = D_QAOA_layer

        file_exists = os.path.isfile(file_name_XY)

        with open(file_name_XY, mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Size", "Depth", "qc Depth"])
            
            writer.writerow([size, depth, depth_qc])
            file.flush()

        print(layer_df.to_string(index=False))  
        print(f"\n Estimated depth of H_C: {depth}")
        print(f" Number of qubits: {size}")
        # print(f" Number of ZZ layers: {num_ZZ_layers}")
        # print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
        print(f" Estimated depth of one QAOA layer: {D_QAOA_layer}")
#         ########################### PENALTY ############################

        from itertools import combinations

        # add penalty terms to the matrix so as to discourage the selection of two rotamers on the same residue - implementation of the Hammings constraint
        def add_penalty_term(M, penalty_constant, residue_pairs):
            for i, j in residue_pairs:
                M[i][j] += penalty_constant
                M[j][i] += penalty_constant 
            return M

        def generate_pairs(N, rot):
            pairs = []
            for i in range(0, rot * N, rot):
                # Generate all unique pairs within each num_rot-sized group
                pairs.extend((i + a, i + b) for a, b in combinations(range(rot), 2))
            
            return pairs

        P = 1.5
        pairs = generate_pairs(N, rot)
        M = deepcopy(H)
        M = add_penalty_term(M, P, pairs)

        print("Modified Hamiltonian with Penalties:\n", M)

        k = 0
        for i in range(num_qubits):
            k += 0.5 * q[i]

        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    k += 0.5 * 0.25 * Q[i][j]


        q_hamiltonian_pen = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if M[i][j] != 0:
                    pauli = generate_pauli_zij(num_qubits, i, j)
                    op = SparsePauliOp(pauli, coeffs=[M[i][j]])
                    q_hamiltonian_pen += op

        for i in range(num_qubits):
            pauli = generate_pauli_zij(num_qubits, i, i)
            Z_i = SparsePauliOp(pauli, coeffs=[M[i][i]])
            q_hamiltonian_pen += Z_i

        def format_sparsepauliop(op):
            terms = []
            labels = [pauli.to_label() for pauli in op.paulis]
            coeffs = op.coeffs
            for label, coeff in zip(labels, coeffs):
                terms.append(f"{coeff:.10f} * {label}")
            return '\n'.join(terms)

        # print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))


        mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))

        # file_name = f"RESULTS/qH_depths/total_cnots_{rot}rots_penalties.csv"
        file_name_pen = f"RESULTS/qH_depths/total_depth_{rot}rots_penalties.csv"

        # Run the depth analysis with graph coloring
        D_HC, D_QAOA_layer, total_cnot, layer_assignments, details = compute_qaoa_depth(q_hamiltonian_pen, mixer_op, "X")
        # D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(q_hamiltonian)

        # Convert to DataFrame for better readability
        layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
        layer_df = layer_df.sort_values(by="Assigned Layer")

        size = num_qubits
        depth = D_QAOA_layer

        qc_pen = build_qaoa_circuit(layer_assignments, q_hamiltonian_pen, mixer_type="X", reps=1)
        depth_qc = qc_pen.depth()

        file_exists = os.path.isfile(file_name_pen)

        with open(file_name_pen, mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Size", "Depth", "qc Depth"])
            
            writer.writerow([size, depth, depth_qc])
            file.flush()

        # Print results
        print(layer_df.to_string(index=False))  # Display the commuting layers
        print(f"\n Estimated depth of H_C: {depth}")
        print(f" Number of qubits: {size}")
        # print(f" Number of ZZ layers: {num_ZZ_layers}")
        # print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
        print(f" Estimated depth of one QAOA layer: {D_QAOA_layer}")


        ########################### BASELINE ############################

        q_hamiltonian_bl = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if H[i][j] != 0:
                    pauli = generate_pauli_zij(num_qubits, i, j)
                    op = SparsePauliOp(pauli, coeffs=[H[i][j]])
                    q_hamiltonian_bl += op

        for i in range(num_qubits):
            pauli = generate_pauli_zij(num_qubits, i, i)
            Z_i = SparsePauliOp(pauli, coeffs=[H[i][i]])
            q_hamiltonian_bl += Z_i

        def format_sparsepauliop(op):
            terms = []
            labels = [pauli.to_label() for pauli in op.paulis]
            coeffs = op.coeffs
            for label, coeff in zip(labels, coeffs):
                terms.append(f"{coeff:.10f} * {label}")
            return '\n'.join(terms)

        print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian_bl))


        mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))

        # file_name = f"RESULTS/qH_depths/total_cnots_{rot}rots_baseline.csv"
        file_name_bl = f"RESULTS/qH_depths/total_depth_{rot}rots_baseline.csv"

        # Run the depth analysis with graph coloring
        D_HC, D_QAOA_layer, total_cnot, layer_assignments, details = compute_qaoa_depth(q_hamiltonian_bl, mixer_op, "X")
        # D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(q_hamiltonian)

        # Convert to DataFrame for better readability
        layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
        layer_df = layer_df.sort_values(by="Assigned Layer")

        size = num_qubits
        depth = D_QAOA_layer

        qc_bl = build_qaoa_circuit(layer_assignments, q_hamiltonian_bl, mixer_type="X", reps=1)
        depth_qc = qc_bl.depth()

        file_exists = os.path.isfile(file_name_bl)

        with open(file_name_bl, mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Size", "Depth", "qc Depth"])
            
            writer.writerow([size, depth, depth_qc])
            file.flush()

        # Print results
        print(layer_df.to_string(index=False))  # Display the commuting layers
        print(f"\n Estimated depth of H_C: {depth}")
        print(f" Number of qubits: {size}")
        # print(f" Number of ZZ layers: {num_ZZ_layers}")
        # print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
        print(f" Estimated depth of one QAOA layer: {D_QAOA_layer}")


        # def count_cnot_in_initial_state(res, rot):
        #     cnot_per_agate = 3
        #     agates_per_residue = rot - 1
        #     total_cnots = res * agates_per_residue * cnot_per_agate
        #     return total_cnots

        # # Test for several system sizes

        # file_name_init = f"RESULTS/qH_depths/total_cnots_{rot}rots_state_prep.csv"

        # total_cnots = count_cnot_in_initial_state(res, rot)
        # num_qubits = res * rot
        # print(f"🧬 System: {num_qubits} qubits ({res} res, {rot} rot) → CNOTs: {total_cnots}")

        # file_exists = os.path.isfile(file_name_init)


        # with open(file_name_init, mode="a", newline="") as file:
        #     writer = csv.writer(file)

        #     # Write the header only if the file is new
        #     if not file_exists:
        #         writer.writerow(["Size", "CNOTs"])
            
        #     # Append the new result
        #     writer.writerow([size, total_cnots])