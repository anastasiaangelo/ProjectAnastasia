import pyrosetta; pyrosetta.init()
import os

pose = pyrosetta.io.pose_from_pdb("inputs/5tj3.pdb")

#running whole protocols via RosettaScriptsParser
import rosetta.protocols.rosetta_scripts; pyrosetta.init('-no_fconfig @inputs/rabd/common')
from rosetta.protocols.rosetta_scripts import RosettaScriptsParser
pose = pyrosetta.io.pose_from_pdb("inputs/rabd/my_ab.pdb")
original_pose = pose.clone()

parser = RosettaScriptsParser()
protocol = parser.generate_mover("inputs/min_L1.xml")

if not os.getenv("DEBUG"):
     protocol.apply(pose)

#running via XMLObjects and strings
from rosetta.protocols.rosetta_scripts import XmlObjects


pose = original_pose.clone()

min_L1 = """
<ROSETTASCRIPTS>
	<SCOREFXNS>
	</SCOREFXNS>
	<RESIDUE_SELECTORS>
		<CDR name="L1" cdrs="L1"/>
	</RESIDUE_SELECTORS>
	<MOVE_MAP_FACTORIES>
		<MoveMapFactory name="movemap_L1" bb="0" chi="0">
			<Backbone residue_selector="L1" />
			<Chi residue_selector="L1" />
		</MoveMapFactory>
	</MOVE_MAP_FACTORIES>
	<SIMPLE_METRICS>
		<TimingProfileMetric name="timing" />
		<SelectedResiduesMetric name="rosetta_sele" residue_selector="L1" rosetta_numbering="1"/>
		<SelectedResiduesPyMOLMetric name="pymol_selection" residue_selector="L1" />
		<SequenceMetric name="sequence" residue_selector="L1" />
		<SecondaryStructureMetric name="ss" residue_selector="L1" />
	</SIMPLE_METRICS>
	<MOVERS>
		<MinMover name="min_mover" movemap_factory="movemap_L1" tolerance=".1" /> 
		<RunSimpleMetrics name="run_metrics1" metrics="pymol_selection,sequence,ss,rosetta_sele" prefix="m1_" />
		<RunSimpleMetrics name="run_metrics2" metrics="timing,ss" prefix="m2_" />
	</MOVERS>
	<PROTOCOLS>
		<Add mover_name="run_metrics1"/>
		<Add mover_name="min_mover" />
		<Add mover_name="run_metrics2" />
	</PROTOCOLS>
</ROSETTASCRIPTS>
"""


xml = XmlObjects.create_from_string(min_L1)
protocol = xml.get_mover("ParsedProtocol")

if not os.getenv("DEBUG"):
    protocol.apply(pose)

#constructing rosetta objects using XMLObjects
#pulling from whole script
L1_sele = xml.get_residue_selector("L1")
L1_res = L1_sele.apply(pose)
for i in range(1, len(L1_res)+1):
    if L1_res[i]:
        print("L1 Residue: ", pose.pdb_info().pose2pdb(i), ":", i )

#constructing from single section
L1_sele = XmlObjects.static_get_residue_selector('<CDR name="L1" cdrs="L1"/>')
L1_res = L1_sele.apply(pose)
for i in range(1, len(L1_res)+1):
    if L1_res[i]:
        print("L1 Residue: ", pose.pdb_info().pose2pdb(i), ":", i )