from pyrosetta import *
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.core.pack import create_packer_graph
import os

init(extra_options='-ex1 -ex2')

pose = pose_from_pdb("input_files/tyr_protein.pdb")
residue_index = 4
scorefxn = get_score_function()

# Task setup
task = TaskFactory.create_packer_task(pose)
task.restrict_to_repacking()
pose.update_residue_neighbors()
scorefxn.setup_for_packing(pose, task.designing_residues(), task.designing_residues())

# Rotamer sets
rotsets = RotamerSets()
rotsets.set_task(task)
packer_graph = create_packer_graph(pose, scorefxn, task)
rotsets.build_rotamers(pose, scorefxn, packer_graph)
rotamer_set = rotsets.rotamer_set_for_residue(residue_index)

os.makedirs("tyr_rotamers", exist_ok=True)

# Instead of dumping rotamer directly, APPLY it to the pose first
for i in range(1, rotamer_set.num_rotamers() + 1):
    rotamer = rotamer_set.rotamer(i)
    trial_pose = pose.clone()
    trial_pose.replace_residue(residue_index, rotamer.clone(), orient_backbone=False)
    
    # Now extract the residue with ring fully resolved
    tyr_pose = Pose()
    tyr_pose.append_residue_by_jump(trial_pose.residue(residue_index).clone(), 1)
    tyr_pose.dump_pdb(f"tyr_rotamers/tyr_rotamer_{i}.pdb")

print(f"✅ {rotamer_set.num_rotamers()} TYR rotamers saved with full rings")
