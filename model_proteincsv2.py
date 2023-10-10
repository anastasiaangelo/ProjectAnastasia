import pyrosetta; pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *

import csv
from pyrosetta.rosetta.core.scoring import EMapVector
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSet, RotamerSets
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.task import *

import sys
import numpy as np


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

pose.update_residue_neighbors()
sfxn.setup_for_packing(pose, task_pack.designing_residues(), task_pack.designing_residues())


#Analyse energy between residues
#to isolate the contribution from particular pairs of residues
# emap = EMapVector()
n_rots_I = rotsets.nrotamers_for_moltenres(1) # calculating the number of rotamers for residue 1
n_rots_J = rotsets.nrotamers_for_moltenres(2)
E = np.zeros((n_rots_I, n_rots_J))

output_file = "score_summary.csv"



with open(output_file, "w") as f:
    for residue_i in range(1, residue_count + 1):
        rotamer_set_i = RotamerSet(pose, residue_i)

        for rotamer_i in range(rotamer_set_i.num_rotamers()):
            rotamer_set_i.set_rotamer(rotamer_i)

            for residue_j in range(1, residue_count + 1):
                if residue_i != residue_j:
                    rotamer_set_j = RotamerSet(residue_j)

                    for rotamer_j in range(rotamer_set_j.num_rotamers()):
                        rotamer_set_j.set_rotamer(rotamer_j)

                        interaction_energy = pose.energies().onebody_energies(residue_i) + pose.energies().onebody_energies(residue_j)
                        interaction_energy += pose.energies().two_body_energy(residue_i, residue_j)

                        E[rotamer_i, rotamer_j] = interaction_energy
                    
        print("Interaction energy between rotamers of residue 1 and 2:", E[rotamer_i, rotamer_j])
        # print(emap)
        # print(emap[fa_atr]) 
        # print(emap[fa_rep]) 
        # print(emap[fa_sol])
        # f.write(f"Score Interactions between residue {residue_number} : {residue1.name3()} and residue {residue_number+1} : {residue2.name3()} --->> Vdw attractive term: {emap[fa_atr]:.2f} Vdw repulsive term: {emap[fa_rep]:.2f} Solvation term: {emap[fa_sol]:.2f} \n\n\n")





