import pyrosetta; pyrosetta.init()
# Modifica inutile
from pyrosetta.teaching import *
from pyrosetta import *
init()
import csv
from pyrosetta.rosetta.core.scoring import EMapVector

#Using the protein Ras (PDB 6q21)
pose = pyrosetta.pose_from_pdb("inputs/6Q21_A.pdb")
residue_count = pose.total_residue()

#Function to check for hydrogen atoms in a Pose
def has_hydrogen_atoms(pose):
    for residue in pose.residues:
        for atom in residue.atoms():
            atom_protocol = pyrosetta.rosetta.core.chemical.Atom()
            if atom_protocol.is_hydrogen():
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

#Analyse energy between residues
#to isolate the contribution from particular pairs of residues
emap = EMapVector()
output_file = "score_summary.txt"

with open(output_file, "w") as f:
    for residue_number in range(1, residue_count + 1):
        residue1 = pose.residue(residue_number)
        if residue_number == residue_count:
            break
        residue2 = pose.residue(residue_number + 1)
        sfxn.eval_ci_2b(residue1, residue2, pose, emap)
        print("Interaction score values of", residue1.name3(), "and", residue2.name3())
        print(emap)
        print(emap[fa_atr]) 
        print(emap[fa_rep]) 
        print(emap[fa_sol])
        f.write(f"Score Interaction between {residue1.name3()} and {residue2.name3()} --->> vdw attractive term {emap[fa_atr]:.2f} vdw repulsive term {emap[fa_rep]:.2f} solvation term {emap[fa_sol]:.2f} \n\n\n")
        emap.zero() #reset emap vector, recalibrate






