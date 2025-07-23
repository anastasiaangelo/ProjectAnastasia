# %%
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

from qiskit import QuantumCircuit
# %%
# Initiate structure, scorefunction, change PDB files
for n in range(6,8):  # from 2 to 7 inclusive
    res = n
    rot = n

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
    #  only for 2 rotamers
    # def create_xy_hamiltonian(num_qubits):
    #     hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])  
    #     for i in range(0, num_qubits, 2):
    #         if i + 1 < num_qubits:
    #             xx_term = ['I'] * num_qubits
    #             yy_term = ['I'] * num_qubits
    #             xx_term[i] = 'X'
    #             xx_term[i+1] = 'X'
    #             yy_term[i] = 'Y'
    #             yy_term[i+1] = 'Y'
    #             xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
    #             yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])
    #             hamiltonian += xx_op + yy_op
    #     return -hamiltonian 
    
    from qiskit.quantum_info import SparsePauliOp, Pauli

    # swapping between nearest neighbours, 1d chain
    def create_local_XY_mixer(num_qubits, block_size):
        if num_qubits % block_size != 0:
            raise ValueError("num_qubits must be divisible by block_size.")

        hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
        num_blocks = num_qubits // block_size

        for i in range(num_blocks):  # block index
            block_start = i * block_size
            for j in range(0, block_size - 1):  # j = 0 to n-2
                qubit_1 = block_start + j
                qubit_2 = block_start + j + 1

                xx_term = ['I'] * num_qubits
                yy_term = ['I'] * num_qubits

                xx_term[qubit_1] = 'X'
                xx_term[qubit_2] = 'X'
                yy_term[qubit_1] = 'Y'
                yy_term[qubit_2] = 'Y'

                xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
                yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])

                hamiltonian += xx_op + yy_op

        return hamiltonian
    
    # with modulo periodic boundary conditions, all interactions
    def get_XY_mixer(num_qubits, num_rot, transverse_field=1):
        if num_rot < 2:
            raise ValueError("num_rot must be at least 2.")

        hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])

        for i in range(0, num_qubits - num_rot + 1, num_rot):          
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
    
    # with periodic boundary conditions, just nearest neighbour
    def create_periodic_local_xy_mixer(num_qubits, block_size):
        if num_qubits % block_size != 0:
            raise ValueError("num_qubits must be divisible by block_size.")

        num_blocks = num_qubits // block_size
        paulis = []
        coeffs = []

        for block in range(num_blocks):
            base = block * block_size
            for j in range(block_size):
                q1 = base + j
                q2 = base + (j + 1) % block_size  # modulo for periodicity

                # XX term
                xx = ['I'] * num_qubits
                xx[q1] = 'X'
                xx[q2] = 'X'
                paulis.append(Pauli("".join(xx)))
                coeffs.append(0.5)

                # YY term
                yy = ['I'] * num_qubits
                yy[q1] = 'Y'
                yy[q2] = 'Y'
                paulis.append(Pauli("".join(yy)))
                coeffs.append(0.5)

        return SparsePauliOp(paulis, coeffs=np.array(coeffs))


    XY_mixer = create_periodic_local_xy_mixer(num_qubits, rot)

# %%
# ################################## XY depth ##############################
    file_name_XY = f"RESULTS/qH_depths/cnot_depth_{rot}rots_nopenalty.csv"

    def create_product_state(base_circuit, n):
        num_qubits = base_circuit.num_qubits

        product_circuit = QuantumCircuit(num_qubits * n)

        for i in range(n):
            base_circuit_copy = deepcopy(base_circuit)
            product_circuit.compose(base_circuit_copy, qubits=range(i * num_qubits, (i + 1) * num_qubits), inplace=True)
            
        return product_circuit

    def R_gate(theta=np.pi/4):
        from qiskit.quantum_info import Operator
        qc = QuantumCircuit(1)
        qc.ry(theta*np.pi/2, 0)
        qc.rz(np.pi, 0)
        op = Operator(qc)
        return op

    def A_gate(theta=np.pi/4):
        qc = QuantumCircuit(2)
        qc.cx(1, 0)
        rgate = R_gate(theta=theta)
        rgate_adj  = R_gate().adjoint()
        # apply rgate to qubit 0
        qc.unitary(rgate_adj, [1], label='R')
        qc.cx(0, 1)
        qc.unitary(rgate, [1], label='R')
        qc.cx(1, 0)
        return qc

    def symmetry_preserving_initial_state(num_res, num_rot, theta=np.pi/4):
        if num_rot < 2:
            raise ValueError("num_rot must be at least 2.")

        qc = QuantumCircuit(num_rot)
        qc.x(num_rot // 2)
        agate = A_gate(theta=theta)

        for i in range(num_rot - 1):
            qc.compose(agate, [i, i + 1], inplace=True)

        init_state = create_product_state(qc, num_res)
        return init_state

    qc_initial_state = symmetry_preserving_initial_state(num_res=res, num_rot=rot, theta=np.pi/4)
    qc_initial_state.draw('mpl')
    # %% ############################################ q_hamiltonian connectivity ########################################################################

    def pauli_shares_qubits(pauli1, pauli2):
        """
        Determines if two Pauli strings act non-trivially on any common qubit.
        They must be placed in different layers if they touch the same qubit.
        """
        return any(p1 != 'I' and p2 != 'I' for p1, p2 in zip(pauli1, pauli2))
    
    def is_two_qubit_non_identity(label):
        return sum(p != 'I' for p in label) == 2


    def compute_commuting_layers(hamiltonian):
        """
        Uses graph coloring to compute the exact number of layers for ZZ terms based on qubit overlaps.
        Separates ZZ interaction layers and single-qubit Z layers for more accurate depth estimation.

        Parameters:
        - hamiltonian: SparsePauliOp representing the Hamiltonian.

        Returns:
        - depth_HC: The total depth required for H_C.
        - num_ZZ_layers: The number of parallelizable ZZ layers.
        - num_single_Z_layers: The number of single-qubit Z layers.
        - layer_assignments: Dictionary mapping each term to its assigned layer.
        """
        pauli_labels = [pauli.to_label() for pauli in hamiltonian.paulis]

        # Separate ZZ terms and single-qubit Z terms
        def is_two_qubit_interaction(label):
            return sum(p != 'I' for p in label) == 2

        interaction_terms = [label for label in pauli_labels if is_two_qubit_interaction(label)]
        single_qubit_z_terms = [label for label in pauli_labels if label.count('Z') == 1 and sum(p != 'I' for p in label) == 1]

        # Step 1: Create Conflict Graph (Nodes = ZZ terms, Edges = Shared Qubits)
        G = nx.Graph()
        for term in interaction_terms:
            G.add_node(term)  # Each ZZ term is a node

        for i, term1 in enumerate(interaction_terms):
            for j, term2 in enumerate(interaction_terms):
                if i < j and pauli_shares_qubits(term1, term2):
                    G.add_edge(term1, term2)  # Conflict edge (they share a qubit)

        # Step 2: Solve Graph Coloring Problem (to find minimum layers for ZZ terms)
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")

        # Step 3: Assign ZZ Layers
        num_ZZ_layers = max(coloring.values()) + 1 if coloring else 0
        layer_assignments = {term: layer for term, layer in coloring.items()}

        # Step 4: Assign Single-Qubit Z Layers (all single-qubit Z can be parallel in one layer)
        num_single_Z_layers = 1 if single_qubit_z_terms else 0
        for term in single_qubit_z_terms:
            layer_assignments[term] = num_ZZ_layers  # Put single-qubit Z terms in their own separate layer

        # Compute Corrected Depth: (ZZ layers * 3) + (Single-Z layers * 1)
        cnot_depth = (num_ZZ_layers * 2) + (num_single_Z_layers * 0)

        return cnot_depth, num_ZZ_layers, num_single_Z_layers, layer_assignments
    

    def compute_xy_mixer_layers_blockwise(mixer_hamiltonian: SparsePauliOp, block_size: int):
        """
        Computes the number of parallel XY mixer layers (CNOT depth = 2 × this) for periodic local blocks.
        Assumes terms are only between neighbors within each block.
        """
        import networkx as nx

        pauli_labels = [pauli.to_label() for pauli in mixer_hamiltonian.paulis]
        num_qubits = len(pauli_labels[0])
        if num_qubits % block_size != 0:
            raise ValueError("Number of qubits must be divisible by block_size.")

        num_blocks = num_qubits // block_size
        max_layers_per_block = 0

        for b in range(num_blocks):
            block_start = b * block_size
            edges = []

            # Get only 2-qubit terms in this block
            for label in pauli_labels:
                qubits = [i for i, p in enumerate(label) if p in ('X', 'Y')]
                if len(qubits) == 2 and block_start <= qubits[0] < block_start + block_size:
                    edges.append(tuple(sorted(qubits)))

            # Build line graph and apply greedy edge coloring
            G = nx.Graph()
            G.add_edges_from(edges)
            LG = nx.line_graph(G)
            coloring = nx.coloring.greedy_color(LG, strategy='saturation_largest_first')
            num_layers = max(coloring.values()) + 1 if coloring else 0
            max_layers_per_block = max(max_layers_per_block, num_layers)

        cnot_depth = 2 * max_layers_per_block
        return cnot_depth, max_layers_per_block


    def compute_qaoa_depth(cost_hamiltonian, mixer_hamiltonian=None, mixer_type="X"):
        """
        Computes the estimated depth of a full QAOA layer.

        Parameters:
        - cost_hamiltonian: SparsePauliOp for H_C.
        - mixer_hamiltonian: SparsePauliOp for H_B (if XY).
        - mixer_type: "X" or "XY".

        Returns:
        - D_HC: depth of the cost Hamiltonian circuit.
        - D_QAOA_layer: total depth of one QAOA layer.
        - details: dictionary with breakdown of depths and layers.
        """

        # Cost Hamiltonian depth
        D_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(cost_hamiltonian)

        if mixer_type == "X":
            D_HB = 1
            num_XY_layers = 0
        elif mixer_type == "XY":
            if mixer_hamiltonian is None:
                raise ValueError("XY mixer selected but no mixer Hamiltonian provided.")
            cnot_depth, num_XY_layers = compute_xy_mixer_layers_blockwise(mixer_hamiltonian, rot)
        else:
            raise ValueError("Unknown mixer type.")

        ZZ_layers = num_XY_layers + num_ZZ_layers

        cnot_layers = 2 * ZZ_layers

        details = {
            "num_ZZ_layers": num_ZZ_layers,
            "num_single_Z_layers": num_single_Z_layers,
            "num_mixer_layers": num_XY_layers,
            "mixer_type": mixer_type,
        }

        return cnot_layers, ZZ_layers, num_XY_layers, details


    def count_two_qubit_layers(qc):
        return qc.depth(filter_function=lambda gate: gate[0].num_qubits == 2)
    

    def visualize_qubit_interactions(hamiltonian, num_qubits):
        """
        Creates a graph visualization of qubit interactions based on ZZ terms in the Hamiltonian.

        Parameters:
        - hamiltonian: SparsePauliOp representing the Hamiltonian.
        - num_qubits: Number of qubits in the system.

        Returns:
        - A plotted graph showing qubit connectivity.
        """
        pauli_labels = [pauli.to_label() for pauli in hamiltonian.paulis]

        # Initialize graph
        G = nx.Graph()

        # Add nodes for qubits
        G.add_nodes_from(range(num_qubits))

        # Identify ZZ interactions and add edges
        for label in pauli_labels:
            if label.count('Z') == 2:  # Identify ZZ terms
                qubits = [i for i, pauli in enumerate(label) if pauli == 'Z']
                if len(qubits) == 2:
                    G.add_edge(qubits[0], qubits[1])

        # Plot the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)  # Layout for better visualization
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Qubit Interaction Graph from ZZ Terms in Hamiltonian")
        plt.show()


    # Run the depth analysis with graph coloring
    cnot_layers, ZZ_layers, num_mixer_layers, details = compute_qaoa_depth(q_hamiltonian_XY, XY_mixer, "XY")
    depth_HC, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(q_hamiltonian_XY)

    # Run the depth analysis with graph coloring
    zz_layers_init_state = count_two_qubit_layers(qc_initial_state)

    # Convert to DataFrame for better readability
    layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
    layer_df = layer_df.sort_values(by="Assigned Layer")

    size = num_qubits
    depth = cnot_layers

    file_exists = os.path.isfile(file_name_XY)

    with open(file_name_XY, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(["Size", "CNOT Layers", "ZZ Layers Init State"])
        
        # Append the new result
        writer.writerow([size, cnot_layers, zz_layers_init_state])

    # Print results
    print("XY QAOA")
    print(layer_df.to_string(index=False))  # Display the commuting layers
    print(f" Number of qubits: {size}")
    print(f" Number of ZZ layers in cost: {num_ZZ_layers}")
    print(f" Number of total two qubit layers: {ZZ_layers}")
    print(f" Total number of cnot layers: {cnot_layers}")
    print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
    print(f" Number of CNOT mixer layers: {2 * num_mixer_layers}")

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
    
    def generate_initial_bitstring(num_qubits, rot):
        bitstring = ['0'] * num_qubits
        for i in range(0, num_qubits, rot):
            bitstring[i] = '1'  # Set the first bit of each rotamer group
        return ''.join(bitstring)
    
    initial_bitstring = generate_initial_bitstring(num_qubits, rot)
    qc_bl_init = QuantumCircuit(num_qubits)
    for i, bit in enumerate(initial_bitstring):
        if bit == '1':
            qc_bl_init.x(i)

    # print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))

    mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
    file_name_pen = f"RESULTS/qH_depths/cnot_depth_{rot}rots_penalties.csv"

    cnot_layers, ZZ_layers, num_mixer_layers, details = compute_qaoa_depth(q_hamiltonian_pen, mixer_op, "X")
    D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(q_hamiltonian_pen)

    zz_layers_init_state = count_two_qubit_layers(qc_bl_init)

    size = num_qubits
    depth = D_QAOA_layer

    file_exists = os.path.isfile(file_name_pen)

    with open(file_name_pen, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(["Size", "CNOT Layers", "ZZ Layers Init State"])
        
        # Append the new result
        writer.writerow([size, cnot_layers, zz_layers_init_state])

    # Convert to DataFrame for better readability
    layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
    layer_df = layer_df.sort_values(by="Assigned Layer")

    # Print results
    print(layer_df.to_string(index=False))  # Display the commuting layers
    print("Penalty QAOA")
    print(f" Number of qubits: {size}")
    print(f" Number of ZZ layers in cost: {num_ZZ_layers}")
    print(f" Number of total two qubit layers: {ZZ_layers}")
    print(f" Total number of cnot layers: {cnot_layers}")
    print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
    print(f" Number of CNOT mixer layers: {2 * num_mixer_layers}")


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

    file_name_bl = f"RESULTS/qH_depths/cnot_depth_{rot}rots_baseline.csv"
    
    # Run the depth analysis with graph coloring
    cnot_layers, ZZ_layers, num_mixer_layers, details = compute_qaoa_depth(q_hamiltonian_bl, mixer_op, "X")
    D_QAOA_layer, num_ZZ_layers, num_single_Z_layers, layer_assignments = compute_commuting_layers(q_hamiltonian_bl)
    
    file_exists = os.path.isfile(file_name_bl)

    with open(file_name_bl, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(["Size", "CNOT Layers", "ZZ Layers Init State"])
        
        # Append the new result
        writer.writerow([size, cnot_layers, zz_layers_init_state])
    
    # Convert to DataFrame for better readability
    layer_df= pd.DataFrame(list(layer_assignments.items()), columns=["Term", "Assigned Layer"])
    layer_df = layer_df.sort_values(by="Assigned Layer")

    # Print results
    print(layer_df.to_string(index=False))  # Display the commuting layers
    print("BASELINE")
    print(f" Number of qubits: {size}")
    print(f" Number of ZZ layers in cost: {num_ZZ_layers}")
    print(f" Number of total two qubit layers: {ZZ_layers}")
    print(f" Total number of cnot layers: {cnot_layers}")
    print(f" Number of single-qubit Z layers: {num_single_Z_layers}")
    print(f" Number of CNOT mixer layers: {2 * num_mixer_layers}")

# %%
