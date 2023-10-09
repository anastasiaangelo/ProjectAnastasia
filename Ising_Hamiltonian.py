# Now contrsuct the Hamiltonian for our problem using the interaction energies previously calculated
import pyrosetta; pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *
init()

from pyrosetta.rosetta.core.scoring import EMapVector
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
import csv

pose = pyrosetta.pose_from_pdb("inputs/6Q21_A.pdb")
residue_count = pose.total_residue()
sfxn = get_score_function(True)
relax_protocol = pyrosetta.rosetta.protocols.relax.FastRelax()
relax_protocol.set_scorefxn(sfxn)
relax_protocol.apply(pose)

Dim = residue_count + 1

Hamiltonian = np.zeros((Dim, Dim))

def spin_up():
    return +1

def spin_down():
    return -1


emap = EMapVector()
output_file = "hamiltonian_terms.csv"
Jii_terms = "diag_terms_ham.csv"

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