import pyrosetta; pyrosetta.init()
from pyrosetta.teaching import *
from pyrosetta import *

from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets, RotamerSet
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack.task import *

import csv
import sys
import numpy as np
import pandas as pd


#Using the protein Ras (PDB 6Q21)
pose = pyrosetta.pose_from_pdb("6Q21_A.pdb")
residue_count = pose.total_residue() # N residues

#Function to check for hydrogen atoms in a Pose
def has_hydrogen_atoms(pose):
    for i in range(1, pose.total_residue() + 1):
        for j in range(1, pose.residue(i).natoms() + 1):
            if pose.residue(i).atom_name(j)[0] == 'H':
                return True
    return False

#Check if the pose has hydrogen atoms
if has_hydrogen_atoms(pose):
    print("Hydrogen atoms are present in the pose.")
else:
    print("No hydrogen atoms found in the pose.")

#Define a score function with the default weights ref2015
sfxn = get_score_function(True)

#Relax the structure to refine it 
relax_protocol = pyrosetta.rosetta.protocols.relax.FastRelax()
relax_protocol.set_scorefxn(sfxn)
# relax_protocol.constrain_relax_to_start_coords(True)
# relax_protocol.set_add_hydrogens(True)
relax_protocol.apply(pose)

if has_hydrogen_atoms(pose):
    print("Now hydrogen atoms are present in the pose.")
else:
    print("still no hydrogens!!")

sfxn.show(pose)

#Print the energy values of each residue
for i in range(1, residue_count + 1): 
    print(pose.energies().show(i))



#For each resdiue now I want to calculate each rotamer
task_pack = TaskFactory.create_packer_task(pose) # is a class in PyRosetta that provides methods for creating and manipulating various types of tasks that control how certain operations are performed on a protein structure (pose).
            # A packer task is a task that defines which residues in a protein structure (pose) are allowed to move or be optimized during a packing step of a computational protein design or refinement protocol.
rotsets = RotamerSets()

#Set up and calculate pairwise interaction energies between rotamers of two specific residues (1 and 2) 
pose.update_residue_neighbors() # updates the neighbour information for residues in the proetein structure
sfxn.setup_for_packing(pose, task_pack.designing_residues(), task_pack.designing_residues()) # prepares the scoring function for packing by specifying which residues are allowed to be designed (optimized) based on the task pack, designing_residues() returns the list of residues that are designated as designable in the task pack.
# packer_task = pyrosetta.rosetta.core.pack.task.PackerTask(pose.total_residue())
# packer_graph = pyrosetta.rosetta.core.pack.creat_packer_graph(pose, packer_task)
packer_neighbor_graph = pyrosetta.rosetta.core.pack.create_packer_graph(pose, sfxn, task_pack) #The neighbor graph represents the spatial relationships between residues and is essential for efficient energy calculations.
rotsets.set_task(task_pack) #associates the rotamer sets with the task pack, indicating which residues and rotamers will be considered during the analysis.
rotsets.build_rotamers(pose, sfxn, packer_neighbor_graph) #builds rotamers for all the protein's residues. Rotamers are alternative conformations of side chains that are sampled during protein modeling.
rotsets.prepare_sets_for_packing(pose, sfxn) #prepares the rotamer sets for packing calculations by potentially reducing the set of rotamers based on their energies.
ig = InteractionGraphFactory.create_interaction_graph(task_pack, rotsets, pose, sfxn, packer_neighbor_graph)
print("built", rotsets.nrotamers(), "rotamers at", rotsets.nmoltenres(), "positions.")
rotsets.compute_energies(pose, sfxn, packer_neighbor_graph, ig, 1) #This computes energies associated with the rotamers in the context of the protein structure. 


#Loop that calculates the pairwise interaction energies between different rotamers (s_i and s_j) at positions 1 and 2 in the protein. The energies are stored in the E 

#Analyse energy between residues
#to isolate the contribution from particular pairs of residues

max_rotamers = 0
for residue_number in range(1, residue_count):
    n_rots = rotsets.nrotamers_for_moltenres(residue_number)
    if n_rots > max_rotamers:
        max_rotamers = n_rots


E = np.zeros((max_rotamers, max_rotamers))

output_file = "score_summary2.csv"
data_list = []
df = pd.DataFrame(columns=['res i', 'res j', 'rot A_i', 'rot B_j', 'E_ij'])


#using panda to save file better

for residue_i in range(1, residue_count):
    rotamer_set_i = rotsets.rotamer_set_for_residue(residue_i) # to access the rotamers generated before (line 69) to calculate the pairwise interaction energies
    molten_res_i = rotsets.resid_2_moltenres(residue_i)
    if rotamer_set_i == None: # skip if no rotamers for the residue
        continue

    for rotamer_i in range(1, rotamer_set_i.num_rotamers() + 1):
        residue_j = residue_i + 1
        if residue_i != residue_j:
            rotamer_set_j = rotsets.rotamer_set_for_residue(residue_j)
            molten_res_j = rotsets.resid_2_moltenres(residue_j)
            if rotamer_set_j == None:
                continue

            for rotamer_j in range(1, rotamer_set_j.num_rotamers() + 1):
                # interaction_energy = pose.energies().onebody_energies(residue_i)[rotamer_i] + pose.energies().onebody_energies(residue_j)[rotamer_j]
                # interaction_energy += pose.energies().two_body_energy(residue_i, residue_j)[rotamer_i][rotamer_j]

                # E[rotamer_i-1, rotamer_j-1] = interaction_energy
                E[rotamer_i-1, rotamer_j-1] = ig.get_two_body_energy_for_edge(molten_res_i, molten_res_j, rotamer_i, rotamer_j)

    # to print and save interactions for each pair of rotamers acrss the residues, not just the last pair
    for rotamer_i in range(1, rotamer_set_i.num_rotamers() + 1):
        for rotamer_j in range(1, rotamer_set_j.num_rotamers() + 1):  
            print(f"Interaction energy between rotamers of residue {residue_i} (rotamer {rotamer_i}) and residue {residue_j} (rotamer {rotamer_j}): {E[rotamer_i-1, rotamer_j-1]}")
            #f.write(f"Score Interactions between residue {residue_i} (rotamer {rotamer_i}) and residue {residue_j} (rotamer {rotamer_j}) --->> {E[rotamer_i-1, rotamer_j-1]}\n")
            data = {'res i': residue_i, 'res j': residue_j, 'rot A_i': rotamer_i, 'rot B_j': rotamer_j, 'E_ij': E[rotamer_i-1, rotamer_j-1]}
            data_list.append(data)

            
df = pd.DataFrame(data_list)
df.to_csv('score_summary2.csv', index=False)




