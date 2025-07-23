# %%
import pyrosetta
pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *

import csv
import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta import PyMOLMover

import threading, time, psutil, os

def memory_monitor(interval=1):
    pid = os.getpid()
    process = psutil.Process(pid)
    while True:
        mem = process.memory_info().rss / (1024 ** 2)  # in MB
        print(f"[Memory Monitor] RSS Memory: {mem:.2f} MB")
        time.sleep(interval)

# Start the monitor before your heavy code
# threading.Thread(target=memory_monitor, daemon=True).start()



# Initiate structure, scorefunction, change PDB files
# num_res = range(7, 8)
# num_rot = range(7, 8)

for n in range(3, 4):  # from 2 to 7 inclusive
    res = n
    rot = n

# for res in num_res:
#     for rot in num_rot:

    ###################################### Set up protein ######################################
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

    import numpy as np
    import pandas as pd
    import csv
    import os
    from copy import deepcopy


    ########################### Configure the hamiltonian from the values calculated classically with pyrosetta ############################
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

    ###################################### Build Quantum Hamiltonian ######################################
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

    terms = []
    coeffs = []

    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if H[i][j] != 0:
                terms.append(generate_pauli_zij(num_qubits, i, j).to_label())
                coeffs.append(H[i][j])

    for i in range(num_qubits):
        terms.append(generate_pauli_zij(num_qubits, i, i).to_label())
        coeffs.append(H[i][i])

    q_hamiltonian = SparsePauliOp.from_list(list(zip(terms, coeffs)))

    def format_sparsepauliop(op):
        terms = []
        labels = [pauli.to_label() for pauli in op.paulis]
        coeffs = op.coeffs
        for label, coeff in zip(labels, coeffs):
            terms.append(f"{coeff:.10f} * {label}")
        return '\n'.join(terms)

    print(f"\nThe hamiltonian constructed using Pauli operators is: \n", format_sparsepauliop(q_hamiltonian))
    
    ###################################### Quantum Optimisation XY ######################################
    import networkx as nx
    import time
    from qiskit import QuantumCircuit

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

    start_time = time.time()
    p = 1
    initial_point = np.ones(2 * p)

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

    from qiskit.circuit.library import QAOAAnsatz
    from qiskit import transpile, QuantumCircuit, QuantumRegister
    from qiskit.transpiler import CouplingMap, Layout
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_ibm_runtime import QiskitRuntimeService

    # file_path_depth = f"RESULTS/CNOTs/total_cnots_{rot}rots_XY_no_init.csv"
    # file_path_depth = f"RESULTS/CNOTs/cnots_{rot}rots_XY.csv"
    file_path_depth = f"RESULTS/CNOTs/cnots_{rot}rots_XY_opt.csv"

    ansatz = QAOAAnsatz(q_hamiltonian, mixer_operator=XY_mixer, reps=p)
    print('\n\nQAOAAnsatz: ', ansatz)
    
    ansatz.count_ops

    qr = QuantumRegister(num_qubits, 'q')
    trivial_layout = Layout({qr[i]: i for i in range(num_qubits)})
# %%
    routing_method = 'stochastic'
    optimization_level = 3
    layout_method = 'trivial'

    # routing_method = None
    # optimization_level = 0
    # layout_method = None

    torino_coupling_map = FakeTorino().coupling_map

    full_circuit = qc_initial_state.compose(ansatz, front=False)

    from qiskit import qpy

    with open(f"full_circuit_XY_opt_{num_qubits}_qubits.qpy", "wb") as f:
        qpy.dump(full_circuit, f)


    # trivial_layout = Layout({qubit: i for i, qubit in enumerate(full_circuit.qubits)})  

    # Transpile hamiltonian and initial state
    full_circuit_isa = transpile(full_circuit, backend=None, initial_layout=trivial_layout,
                                coupling_map=torino_coupling_map, basis_gates=['cx', 'rz', 'rx', 'id', 'x', 'sx'], 
                                optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method)
    
    print("\n\nAnsatz layout after explicit transpilation:", full_circuit_isa._layout)

    op_counts = full_circuit_isa.count_ops()
    total_gates = sum(op_counts.values())
    two_qubit_gate_names = [name for name, count in op_counts.items() 
                            if any(len(op.qubits) == 2 for op in full_circuit_isa if op.operation.name == name)]
    total_two_qubit_gates = sum(op_counts[name] for name in two_qubit_gate_names)
    depth = full_circuit_isa.depth()
    two_qubit_depth = full_circuit_isa.depth(
        filter_function=lambda instr: len(instr.qubits) == 2
    )
    cnots = op_counts.get('cx', 0)
    cnot_depth = full_circuit_isa.depth(
    filter_function=lambda instr: instr.operation.name == 'cx')

    depth = full_circuit_isa.depth()
    print("Operation counts:", op_counts)
    print("Total number of 2Q gates:", total_two_qubit_gates)
    print("CNOTS:", cnots)
    print("CNOT Depth of the circuit:", cnot_depth)
    print("2Q Depth of the circuit: ", two_qubit_depth)


# %%
    data_depth = {
        "Experiment": ["Hardware XY-QAOA"],
        "CNOTs": [cnots],
        "CNOT Depth": [cnot_depth]
    }

    df_depth = pd.DataFrame(data_depth)

    file_exists = os.path.isfile(file_path_depth) and os.path.getsize(file_path_depth) > 0

    # with open(file_path_depth, mode="a", newline="") as file:
    #     writer = csv.writer(file)

    #     # Write the header only if the file is new
    #     if not file_exists:
    #         writer.writerow(["Size", "CNOTs", "cnot Depth"])
        
    #     # Append the new result
    #     writer.writerow([num_qubits, cnots, cnot_depth])


    try:
        with open(file_path_depth, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Size", "CNOTs", "cnot Depth"])
    except FileExistsError:
        pass  # File already exists, don't overwrite header

    # Append data
    with open(file_path_depth, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_qubits, cnots, cnot_depth])

# %%
    ###################################### Baseline ######################################
    # file_name_baseline = f"RESULTS/CNOTs/total_cnots_{rot}rots_baseline_no_init.csv"
    # file_name_baseline = f"RESULTS/CNOTs/cnots_{rot}rots_baseline.csv"
    file_name_baseline = f"RESULTS/CNOTs/cnots_{rot}rots_baseline_opt.csv"

    mixer_op = sum(X_op(i,num_qubits) for i in range(num_qubits))
    
    ansatz_baseline = QAOAAnsatz(q_hamiltonian, mixer_operator=mixer_op, reps=p)

    def generate_initial_bitstring(num_qubits):
        bitstring = [(i%2) for i in range(num_qubits)]
        return ''.join(map(str, bitstring))
    
    initial_bitstring = generate_initial_bitstring(num_qubits)
    qc_bl_init = QuantumCircuit(num_qubits)
    for i, bit in enumerate(initial_bitstring):
        if bit == '1':
            qc_bl_init.x(i)

    full_circuit_bl = qc_bl_init.compose(ansatz_baseline, front=False)

    with open(f"full_circuit_bl_opt_{num_qubits}_qubits.qpy", "wb") as f:
        qpy.dump(full_circuit_bl, f)

    # trivial_layout = Layout({qubit: i for i, qubit in enumerate(full_circuit_bl.qubits)})  

    full_circuit_bl_isa = transpile(full_circuit_bl, backend=None, initial_layout=trivial_layout,
                                coupling_map=torino_coupling_map, basis_gates=['cx', 'rz', 'rx', 'id', 'x', 'sx'], 
                                optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method)

    print("\n\nAnsatz layout after explicit transpilation:", full_circuit_bl_isa._layout)

    op_counts = full_circuit_bl_isa.count_ops()
    total_gates = sum(op_counts.values())
    two_qubit_gate_names = [name for name, count in op_counts.items() 
                            if any(len(op.qubits) == 2 for op in full_circuit_bl_isa if op.operation.name == name)]
    total_two_qubit_gates = sum(op_counts[name] for name in two_qubit_gate_names)
    depth = full_circuit_bl_isa.depth()
    two_qubit_depth = full_circuit_bl_isa.depth(
        filter_function=lambda instr: len(instr.qubits) == 2
    )

    cnots = op_counts.get('cx', 0)
    cnot_depth = full_circuit_bl_isa.depth(
    filter_function=lambda instr: instr.operation.name == 'cx')

    print("Operation counts:", op_counts)
    print("Total number of gates:", total_gates)
    print("CNOTS:", cnots)
    print("CNOT Depth of the circuit:", cnot_depth)
    print("Depth of the circuit: ", depth)

    data_depth = {
        "Experiment": ["Hardware XY-QAOA"],
        "Total number of gates": [total_gates],
        "Depth of the circuit": [depth],
        "CZs": [total_two_qubit_gates]
    }

    df_depth = pd.DataFrame(data_depth)
    file_exists = os.path.isfile(file_name_baseline)

    try:
        with open(file_name_baseline, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Size", "CNOTs", "cnot Depth"])
    except FileExistsError:
        pass  # File already exists, don't overwrite header

    # Append data
    with open(file_name_baseline, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_qubits, cnots, cnot_depth])


    # with open(file_name_baseline, mode="a", newline="") as file:
    #     writer = csv.writer(file)

    #     # Write the header only if the file is new
    #     if not file_exists:
    #         writer.writerow(["Size", "Depth", "Two-Qubit Gates", "two_qubit Depth", "Total Gates"])
        
    #     # Append the new result
    #     writer.writerow([num_qubits, depth, total_two_qubit_gates, two_qubit_depth, total_gates])

# %%
    ###################################### Penalty ######################################
    # file_name_penalty = f"RESULTS/CNOTs/total_cnots_{rot}rots_penalty_no_init.csv"
    # file_name_penalty = f"RESULTS/CNOTs/cnots_{rot}rots_penalty.csv"
    file_name_penalty = f"RESULTS/CNOTs/cnots_{rot}rots_penalty_opt.csv"

    from itertools import combinations

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

    terms = []
    coeffs = []

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

    # q_hamiltonian_pen = SparsePauliOp.from_list(list(zip(terms, coeffs)))

    ansatz_penalty = QAOAAnsatz(q_hamiltonian_pen, mixer_operator=mixer_op, reps=p)

    full_circuit_pen = qc_bl_init.compose(ansatz_penalty, front=False)

    with open(f"full_circuit_pen_opt_{num_qubits}_qubits.qpy", "wb") as f:
        qpy.dump(full_circuit_pen, f)

    full_circuit_pen_isa = transpile(full_circuit_pen, backend=None, initial_layout=trivial_layout,
                                coupling_map=torino_coupling_map, basis_gates=['cx', 'rz', 'rx', 'id', 'x', 'sx'], 
                                optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method)

    print("\n\nAnsatz layout after explicit transpilation:", full_circuit_pen_isa._layout)
    
    op_counts = full_circuit_pen_isa.count_ops()
    total_gates = sum(op_counts.values())
    two_qubit_gate_names = [name for name, count in op_counts.items() 
                            if any(len(op.qubits) == 2 for op in full_circuit_pen_isa if op.operation.name == name)]
    total_two_qubit_gates = sum(op_counts[name] for name in two_qubit_gate_names)
    depth = full_circuit_pen_isa.depth()
    two_qubit_depth = full_circuit_pen_isa.depth(
        filter_function=lambda instr: len(instr.qubits) == 2
    )

    cnots = op_counts.get('cx', 0)
    cnot_depth = full_circuit_pen_isa.depth(
    filter_function=lambda instr: instr.operation.name == 'cx')

    print("Operation counts:", op_counts)
    print("Total number of gates:", total_gates)
    print("CNOTS:", cnots)
    print("CNOT Depth of the circuit:", cnot_depth)
    print("Depth of the circuit: ", depth)

    data_depth = {
        "Experiment": ["Hardware XY-QAOA"],
        "Total number of gates": [total_gates],
        "Depth of the circuit": [depth],
        "CNOTs": [total_two_qubit_gates]        
        }

    file_exists = os.path.isfile(file_name_penalty)
    df_depth = pd.DataFrame(data_depth)

    # with open(file_name_penalty, mode="a", newline="") as file:
    #     writer = csv.writer(file)

    #     # Write the header only if the file is new
    #     if not file_exists:
    #         writer.writerow(["Size", "Depth", "Two-Qubit Gates", "two_qubit Depth", "Total Gates"])
        
    #     # Append the new result
    #     writer.writerow([num_qubits, depth, total_two_qubit_gates, two_qubit_depth, total_gates])

    try:
        with open(file_name_penalty, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Size", "CNOTs", "cnot Depth"])
    except FileExistsError:
        pass  # File already exists, don't overwrite header

    # Append data
    with open(file_name_penalty, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_qubits, cnots, cnot_depth])

# full_circuit_pen_isa.draw(output='mpl', idle_wires=False)
# plt.show()


    

# %%
