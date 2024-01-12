import pyrosetta
pyrosetta.init()

pose = pyrosetta.io.pose_from_pdb("inputs/5tj3.pdb")
pose.sequence()

from pyrosetta.toolbox import cleanATOM
cleanATOM("inputs/5tj3.pdb")
pose_clean = pyrosetta.io.pose_from_pdb("inputs/5tj3.clean.pdb")
pose_clean.sequence()
pose.annotated_sequence() # to see the differences in more detail
pose_clean.annotated_sequence()
print(pose.total_residue())
residue20 = pose.residue(20) # store the information for residue 20
print(residue20.name())
residue24 = pose.residue(24)

#PSB and pose numbering is different, to convert them
print(pose.pdb_info().chain(24))
print(pose.pdb_info().number(24))
print(pose.pdb_info().pdb2pose('A',24))
print(pose.pdb_info().pose2pdb(1))
res_24 = pose.residue(24)
print(res_24.name())

#to confirm that PyRosetta has loaded the zinc ions as metal ions
zn_resid = pose.pdb_info().pdb2pose('A',601)
res_zn = pose.residue(zn_resid)
res_zn.is_metal()
res_24 = pose.residue(24)
res_24.atom_is_backbone
res_24.atom_is_backbone(res_24.atom_index("CA"))