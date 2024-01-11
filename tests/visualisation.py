import pyrosetta; pyrosetta.init()

pose = pyrosetta.io.pose_from_pdb("test.pdb")

#creating an instance of PyMOLMover pmm
from pyrosetta import PyMOLMover
pmm = PyMOLMover() 
clone_pose = pyrosetta.io.Pose()
clone_pose.assign(pose)
pmm.apply(clone_pose)

pmm.send_hbonds(clone_pose)

pmm.keep_history(True) 
pmm.apply(clone_pose)
clone_pose.set_phi(5, -90)
pmm.apply(clone_pose)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython import display

from pathlib import Path
gifPath = Path("./Media/PyMOL-tutorial.gif")
# Display GIF in Jupyter, CoLab, IPython
with open(gifPath,'rb') as f:
    display.Image(data=f.read(), format='png',width='800')
