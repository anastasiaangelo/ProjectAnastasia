

# GROMACS Molecular Dynamics Simulation Script

# Step 1: Create Topology
gmx pdb2gmx -f test.pdb -o processed.gro -water spce

# Step 2: Define the Simulation Box
gmx editconf -f processed.gro -o newbox.gro -c -d 1.0 -bt cubic

# Step 3: Solvation
gmx solvate -cp newbox.gro -cs spc216.gro -o solvated.gro -p topol.top

# Step 4: Add Ions
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr
echo 'SOL' | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

# Step 5: Energy Minimization
gmx grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# Step 6: Equilibration NVT
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt

# Step 7: Equilibration NPT
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -deffnm npt

# Step 8: Molecular Dynamics Simulation
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr
gmx mdrun -deffnm md_0_1

# Analysis (Example: RMSD Calculation)
gmx rms -s md_0_1.tpr -f md_0_1.xtc -o rmsd.xvg

echo "GROMACS Simulation Completed"
