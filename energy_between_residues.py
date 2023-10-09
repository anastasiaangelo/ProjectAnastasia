#import pyrosettacolabsetup
#pyrosettacolabsetup.installpyrosetta()

import pyrosetta
pyrosetta.init()

from pyrosetta import *
from pyrosetta.teaching import *
init()

sfxn = get_score_function(True)

#Analyze the energy between residues Y102 and Q408 in cetuximab
# get the pose numbers for Y102 (chain D) and Q408 (chain A)

pose = pyrosetta.toolbox.pose_from_rcsb("1YY9")
res102 = pose.pdb_info().pdb2pose("D", 102)
res408 = pose.pdb_info().pdb2pose("A", 408)

#Score the pose and determine the van der Waals energies and solvation energy between these two residues.
# isolate contributions from particular pairs of residues
emap = EMapVector()
sfxn.eval_ci_2b(pose.residue(res102), pose.residue(res408), pose, emap)
print(emap[fa_atr]) #vdw attractive term between the two residues
print(emap[fa_rep]) #vdw repulsive term between the two residues
print(emap[fa_sol]) #solvation energy term between the two residues

print(pose.residue(res102).name3())