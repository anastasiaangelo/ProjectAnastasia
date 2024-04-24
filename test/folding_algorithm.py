import pyrosetta; pyrosetta.init()
from pyrosetta import *
from pyrosetta.teaching import *
init()
import math
import random

# modifica

# Folding a 10 residue protein - using a MonteCarlo algorithm to optimise the protein conformation

# Create a simply poly-alanine pose with 10 residues
polyA = pyrosetta.pose_from_sequence('A'*10)
polyA.pdb_info().name("polyA")

print("phi: %i" %polyA.phi(9))
print("psi: %i" %polyA.psi(9))

pmm = PyMOLMover()
pmm.keep_history(True)

pmm.apply(polyA)

# Choose one residue at random and then randomnly perturb either of the angles by a random number chosen from a Gaussian distribution

def randTrial(your_pose):
    randNum = random.randint(2, your_pose.total_residue())
    currPhi = your_pose.phi(randNum)
    currPsi = your_pose.psi(randNum)
    newPhi = random.gauss(currPhi, 25)
    newPsi = random.gauss(currPsi, 25)
    your_pose.set_phi(randNum, newPhi)
    your_pose.set_psi(randNum, newPsi)
    pmm.apply(your_pose)
    return your_pose

# Scoring move

sfxn = get_fa_scorefxn()

def score(your_pose):
    return sfxn(your_pose)

# Accepting/Rejecting move based on Metropolis criterion which has a probability of accepting a move as P=exp(−ΔG/kT)
# When ΔE≥0, the Metropolis criterion probability of accepting the move is P=exp(−ΔG/kT). When ΔE<0, the Metropolis criterion probability of accepting the move is P=1. Use kT=1 Rosetta Energy Unit (REU).

def decision(before_pose, after_pose):
    E = score(after_pose) - score(before_pose)
    if E < 0:
        return after_pose
    elif random.uniform(0,1) >= math.exp(-E/1):
        return before_pose
    else:
        return after_pose  


# Execution - loop of 100 iterations, making a random trial move, scoring the protein and accepting/rejecting the move
# after each iteration, output the curretn pose energy and the lowesr energy ever observed. The final output should be the lowest energy conformation that is achieved in any point during the simulation

def basic_folding(your_pose):
    lowest_pose = Pose() # create an empty pose to track the lowest energy pose
    for i in range(100):
        if i == 0:
            lowest_pose.assign(your_pose)

        before_pose = Pose()
        before_pose.assign(your_pose) # keep track of pose before random move

        after_pose = Pose()
        after_pose.assign(randTrial(your_pose)) # do random move and store the pose

        your_pose.assign(decision(before_pose, after_pose))

        if score(your_pose) < score(lowest_pose):
            lowest_pose.assign(your_pose)

        print("Iteration # %i" %i)
        print("Current pose score: %1.3f" %score(your_pose))
        print("Lowest pose score: %1.3f" %score(lowest_pose))

    return lowest_pose

basic_folding(polyA)