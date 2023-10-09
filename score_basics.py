import pyrosetta; pyrosetta.init()
import pyrosetta
pyrosetta.init()

ras = pyrosetta.pose_from_pdb("inputs/6Q21_A.pdb")
from pyrosetta.teaching import *

sfxn = get_score_function(True)

print(sfxn)

sfxn2 = ScoreFunction()
sfxn2.set_weight(fa_atr, 1.0)
sfxn2.set_weight(fa_rep, 1.0)

print(sfxn(ras))
print(sfxn2(ras))

sfxn.show(ras)

print(ras.energies().show(24))

res24 = ras.residue(24)
res20 = ras.residue(20)
res24_atomN = res24.atom_index("N")
res20_atomO = res20.atom_index("O")
pyrosetta.etable_atom_pair_energies(res24, res24_atomN, res20, res20_atomO, sfxn)
