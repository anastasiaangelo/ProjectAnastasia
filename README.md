# ProjectAnastasia

**Repository for Protien Folding project**

Create a virtual environment with the packages in requirements.txt, in the location: /venv/lib/python3.10/site-packages/qiskit_algorithms/minimum_eigensolvers/sampling_vqe.py substitute the sampling_vqe.py file found in this github in order to obtain the intermediate bitstrings and energy values of the whole QAOA process.

**1. Ising_Hamiltonian.py**
on line 13 of the code a PDB input file from the folder 'input_files' can be chosen and run in the script, the packing and relaxation is performed based on the number of           rotamers specified on line 89 of the code (num_rot = 2,3,etc.) and two output CSV files are generated: one_body_terms.csv and two_body_terms.csv in the folder 'energy_files'

**2. Optimisation scripts** can now be run which will take the one and two body energy CSV files as input and will write an output file to a RESULTS folder specified in the file_path at the beginning of the script.

_- 2 rotamers per residue:_

**> Ising_nopenalty.py :** performs the brute force optimisation of the 2 rotamer per residue test structure, the first two cells run through all the possible bitstrings                 that respect the Hamming's constraint and select the one with lowest energy, serving as a benchmark.
          
**> Ising_localpenalty_hardware.py :** performs the optimisation of the structure with QAOA, with the Hamiltonian built with the local penalty terms on each pair of                      residues, the cell with the classical optimisation can be run up to a certain amount of qubits for a benchmark after which it becomes too computationlly heavy.                     Then the noisy simualtions with Aer can be run, post selection is implemented considering also the intermediate bitstrings of the QAOA process and the simualtions                  on the hardware in the various cells, for the hardware the coupling map must be modified to fit the specified machine beign used and the number of qubits.
          
**> Ising_mixer.py :** performs the optimisation of the structrue with QAOA with the XY mixer, all the same comments as for the local penalties apply.
           
_- 3 rotamers per residue:_

**> brute_force_3rot.py :** performs the brute force optimisation of the 3 rotamer per residue test structure, the first two cells run through all the possible bitstrings that respect the Hamming's constraint and select the one with lowest energy, serving as a benchmark.
           
**> local_penalty_3rot.py :** performs the optimisation of the 3 rotamer per residue structure with QAOA, with the Hamiltonian built with the local penalty terms on each pair of residues, structured in the same manner as the 2 rotamer per residue script.
           
**> 3rot_XYmixer.py :** performs the optimisation of the structrue with QAOA with the XY mixer for 3 rotamers per residue, once again sturctured identically as the 2                     rotamer equivalent.
