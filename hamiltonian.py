# Program that solves the one-particle Schrodinger equation for a potential specified in function potential()
# This example is for the harmonic oscillator in 3D
import pyrosetta; pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *
init()

import numpy as np
from matplotlib import pyplot as plt

# Function for initialisation of parameters
def initialize():
    Rmin = 0.0
    Rmax = 10.0
    lOrbital = 0
    Dim = 400
    return Rmin, Rmax, lOrbital, Dim

# Set up the HO potential

def potential(r):
    return r*r

# Get the boundary, orbital momentum and number of intergation points
Rmin, Rmax, lOrbital, Dim = initialize()

# Initialise constants
Step = Rmax/(Dim+1)
DiagConst = 2.0/(Step*Step)
NondiagConst = -1.0/(Step*Step)
OrbitalFactor = lOrbital*(lOrbital+1.0)

# Calculate array of potential values
v = np.zeros(Dim)
r = np.linspace(Rmin, Rmax, Dim)
for i in range(Dim):
    r[i] = Rmin + (i+1)*Step;
    v[i] = potential((r[i]) + OrbitalFactor/(r[i]*r[i]));

# Setting up a tridiagonal matrix and finding eigenvectors and eigenvalues
Hamiltonian = np.zeros((Dim,Dim))
Hamiltonian[0,0] = DiagConst + v[0];
Hamiltonian[0,1] = NondiagConst;
for i in range(1,Dim-1):
    Hamiltonian[i,i-1]  = NondiagConst;
    Hamiltonian[i,i]    = DiagConst + v[i];
    Hamiltonian[i,i+1]  = NondiagConst;
Hamiltonian[Dim-1,Dim-2] = NondiagConst;
Hamiltonian[Dim-1,Dim-1] = DiagConst + v[Dim-1];

# Diagonalise and obtain eigenvalues, not necessarily sorted
EigValues, EigVectors = np.linalg.eig(Hamiltonian)

# Sort eigenvectors and eigenvalues
permute = EigValues.argsort()
EigValues = EigValues[permute]
EigVectors = EigVectors[:,permute]

# Plot the results for the three lowest lying eigenstates
for i in range(3):
    print(EigValues[i])
FirstEigvector = EigVectors[:,0]
SecondEigvector = EigVectors[:,1]
ThirdEigvector = EigVectors[:,2]

plt.plot(r, FirstEigvector**2 ,'b-',r, SecondEigvector**2 ,'g-',r, ThirdEigvector**2 ,'r-')
plt.axis([0,4.6,0.0, 0.025])
plt.xlabel(r'$r$')
plt.ylabel(r'Radial probability $r^2|R(r)|^2$')
plt.title(r'Radial probability distributions for three lowest-lying states')
plt.savefig('eigenvector.pdf')
plt.show()