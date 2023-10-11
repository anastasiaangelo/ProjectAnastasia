# Now contrsuct the Hamiltonian for our problem using the interaction energies previously calculated
import pyrosetta; pyrosetta.init()
from pyrosetta.teaching import *
from pyrosetta import *

import csv
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack.task import *

#Initiate structure, scorefunction
pose = pyrosetta.pose_from_pdb("6Q21_A.pdb")
residue_count = pose.total_residue()
sfxn = get_score_function(True)

relax_protocol = pyrosetta.rosetta.protocols.relax.FastRelax()
relax_protocol.set_scorefxn(sfxn)
relax_protocol.apply(pose)

#Define task and interaction graph
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

#Define dimension for matrix
max_rotamers = 0
for residue_number in range(1, residue_count):
    n_rots = rotsets.nrotamers_for_moltenres(residue_number)
    if n_rots > max_rotamers:
        max_rotamers = n_rots

E = np.zeros((max_rotamers, max_rotamers))
Hamiltonian = np.zeros((max_rotamers, max_rotamers))

output_file = "hamiltonian_terms.csv"

def spin_up():
    return +1

def spin_down():
    return -1


#Loop to find hamiltonian values Jij
with open(output_file, "w") as f:
    for residue_number in range(1, residue_count + 1):
        rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
        if rotamer_set_i == None: # skip if no rotamers for the residue
            continue

        for residue_number2 in range(1, residue_count+1):
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
                S1 = spin_up()
                S2 = spin_down()
                E[rot_i-1, rot_j-1] = ig.get_two_body_energy_for_edge(molten_res_i, molten_res_j, rot_i, rot_j)
                Hamiltonian[rot_i-1, rot_j-1] = E[rot_i-1, rot_j-1]*S1*S2
    
        for rot_i in range(1, rotamer_set_i.num_rotamers() + 1):
            for rot_j in range(1, rotamer_set_j.num_rotamers() + 1):
                print(f"Interaction energy between rotamers of residue {residue_number} rotamer {rot_i} and residue {residue_number2} rotamer {rot_j} :", E[rot_i-1, rot_j-1])
                f.write(f"Score Interactions between residue {residue_number} rotamer {rot_i} and residue {residue_number2} rotamer {rot_j} -> {E[rot_i-1, rot_j-1]}\n")

                    

#Loop to find hamiltonian values Jii
with open(output_file, "a", newline='') as f:
    for residue_number in range(1, residue_count + 1):
        residue1 = pose.residue(residue_number)
        rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
        if rotamer_set_i == None: # skip if no rotamers for the residue
            continue

        molten_res_i = rotsets.resid_2_moltenres(residue_number)
        
        for rot_i in range(1, rotamer_set_i.num_rotamers() + 1):
            S1 = spin_up()
            E[rot_i-1, rot_i-1] = ig.get_one_body_energy_for_node_state(molten_res_i, rot_i)
            Hamiltonian[rot_i-1, rot_i-1] = E[rot_i-1, rot_i-1]*S1

            print(f"Interaction score values of {residue1.name3()} rotamer {rot_i} with itself {E[rot_i-1][rot_i-1]}")
            f.write(f"Score Interaction of residue {residue_number} : {residue1.name3()}, rotamer {rot_i} with itself --> {Hamiltonian[rot_i-1, rot_i-1]} \n\n")
        

#Save the Hamiltonian to a csv file
np.savetxt("hamiltonian.csv", Hamiltonian, delimiter=",", fmt="%d")