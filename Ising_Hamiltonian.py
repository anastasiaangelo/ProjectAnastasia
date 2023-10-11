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

#initiate structure, scorefunction
pose = pyrosetta.pose_from_pdb("inputs/6Q21_A.pdb")
residue_count = pose.total_residue()
sfxn = get_score_function(True)

relax_protocol = pyrosetta.rosetta.protocols.relax.FastRelax()
relax_protocol.set_scorefxn(sfxn)
relax_protocol.apply(pose)

#define task and interaction graph
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

#define dimension for matrix
max_rotamers = 0
for residue_number in range(1, residue_count):
    n_rots = rotsets.nrotamers_for_moltenres(residue_number)
    if n_rots > max_rotamers:
        max_rotamers = n_rots

E = np.zeros((max_rotamers, max_rotamers))
Hamiltonian = np.zeros((max_rotamers, max_rotamers))

output_file = "hamiltonian_terms.csv"
Jii_terms = "diag_terms_ham.csv"

def spin_up():
    return +1

def spin_down():
    return -1


# with open(output_file, "r") as f:
#     reader = csv.reader(f)
#     rows = list(reader)

# for row in rows:
#     row.append('new value')


with open(output_file, "w") as f:
    for residue_number in range(1, residue_count + 1):
        residue1 = pose.residue(residue_number)
        S1 = spin_up()
        if residue_number == residue_count:
            break
        residue2 = pose.residue(residue_number + 1)
        S2 = spin_down()
        sfxn.eval_ci_2b(residue1, residue2, pose, emap)
        print("Interaction score values of", residue1.name3(), "and", residue2.name3())
        f.write(f"Score Interaction between residue number {residue_number} : {residue1.name3()} and {residue_number+1} : {residue2.name3()} --> Vdw attractive term: {emap[fa_atr]:.2f} Vdw repulsive term: {emap[fa_rep]:.2f} Solvation term: {emap[fa_sol]:.2f} \n\n")
        Hamiltonian[residue_number, residue_number+1] = emap[fa_atr]*S1*S2
        emap.zero()



with open(output_file, "a", newline='') as f:
    for residue_number in range(1, residue_count + 1):
        residue1 = pose.residue(residue_number)
        S1 = spin_up()
        sfxn.eval_ci_2b(residue1, residue1, pose, emap)
        print("Interaction score values of", residue1, "with itself")
        f.write(f"Score Interaction of residue {residue_number} : {residue1.name3()} with itself --> Vdw attractive term: {emap[fa_atr]:.2f} Vdw repulsive term: {emap[fa_rep]:.2f} olvation term: {emap[fa_sol]:.2f} \n\n")
        Hamiltonian[residue_number,residue_number] = emap[fa_atr]*S1
        emap.zero()


np.savetxt("hamiltonian.csv", Hamiltonian, delimiter="")