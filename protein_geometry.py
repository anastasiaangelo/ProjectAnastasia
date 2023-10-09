import pyrosetta
pyrosetta.init()

from pyrosetta import *
from pyrosetta.teaching import *
init()

pose = pose_from_pdb("inputs/5tj3.pdb")
resid = pose.pdb_info().pdb2pose('A', 28)
print("phi:",pose.phi(resid))
print("psi:",pose.psi(resid))
print("chi1:",pose.chi(1,resid))
conformation = pose.conformation()
res_28 = pose.residue(resid)
N28 = AtomID(res_28.atom_index("N"), resid)

from pyrosetta.rosetta.core.id import AtomID
N28 = AtomID(res_28.atom_index("N"), resid)
CA28 = AtomID(res_28.atom_index("CA"), resid)
C28 = AtomID(res_28.atom_index("C"), resid)
print(N28)
print(pose.conformation().bond_length(N28, CA28))
print(pose.conformation().bond_length(CA28, C28))

from pyrosetta.teaching import*
init()

one_res_seq = "V"
pose_one_res = pose_from_sequence(one_res_req)
print(pose_one_res.sequence())

N_xyz = pose_one_res.residue(1).xyz("N")
CA_xyz = pose_one_res.residue(1).xyz("CA")
C_xyz = pose_one_res.residue(1).xyz("C")
print((CA_xyz - N_xyz).norm())
print((CA_xyz - C_xyz).norm())
angle = pose.conformation().bond_angle(N28, CA28, C28)
print(angle)
import math
angle*180/math.pi