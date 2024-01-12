### Take input pdb, score, repack and extract one and two body energies
import pyrosetta;
pyrosetta.init()
from pyrosetta.teaching import *
from pyrosetta import *

import csv
import sys
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser

pose = pyrosetta.pose_from_pdb("Input Files/test1.pdb")

# print out cartesian coordinates of each atom
def extract_coordinates(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    
    coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.coord
                    coordinates.append((x, y, z))

    return coordinates

pdb_file = 'Input Files/test1.pdb'
coords = extract_coordinates(pdb_file)

for coord in coords:
    print(coord)




def create_rotation_matrix(v1, v2, v3):
    """
    Create a rotation matrix that aligns v1, v2, v3 with the x, y, z axes.
    """
    R = np.array([[v1.x, v2.x, v3.x],
                  [v1.y, v2.y, v3.y],
                  [v1.z, v2.z, v3.z]])
    return R

def transform_to_local_coordinates(pose, residue_number):
    """
    Transform the coordinates of a residue's atoms into a local coordinate system.
    """
    residue = pose.residue(residue_number)
    
    # Define the origin (e.g., alpha carbon)
    origin = residue.xyz("CA")

    # Define other points to create axes (e.g., N and C atoms for backbone)
    axis_x_point = residue.xyz("N")
    axis_y_point = residue.xyz("C")
    
    # Create vectors for the axes
    v1 = axis_x_point - origin
    v2 = axis_y_point - origin
    
    # Create a third vector orthogonal to v1 and v2 for the z-axis
    v3 = v1.cross(v2)

    # Normalize the vectors to create an orthonormal basis
    v1.normalize()
    v2.normalize()
    v3.normalize()

    # Create the rotation matrix
    rotation_matrix = create_rotation_matrix(v1, v2, v3)

    # Transform coordinates of each atom in the residue
    local_coords = []
    for i in range(1, residue.natoms() + 1):
        global_coord = residue.xyz(i)
        transformed_coord = np.dot(rotation_matrix, np.array(global_coord - origin))
        local_coords.append(transformed_coord)

    print(f"Local Coordinates for Residue {residue_number}:")
    for coord in local_coords:
        print(coord)

    return local_coords



def calculate_internal_coordinates(pose, residue_number):

    residue = pose.residue(residue_number)
    
    # Example: Select three atoms to define the internal coordinate system
    atom1 = residue.xyz("N")
    atom2 = residue.xyz("CA")
    atom3 = residue.xyz("C")  
    
    # Calculate vectors
    v1 = atom2 - atom1
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(np.cross(atom3 - atom1, v1), v1)
    v2 /= np.linalg.norm(v2)

    # Calculate bond length, bond angle, and dihedral angle
    bond_length = v1.norm()
    bond_angle = np.arccos(np.dot(v1, v2))
    # Dihedral angle calculation would involve a fourth atom

    return bond_length, bond_angle  # Add dihedral if needed


def transform_to_internal_coordinates(atom_coords, residue_number):
    residue = pose.residue(residue_number)
    atom1 = residue.xyz("N")
    atom2 = residue.xyz("CA")
    atom3 = residue.xyz("C") 

    v1 = atom2 - atom1
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(np.cross(atom3 - atom1, v1), v1)
    v2 /= np.linalg.norm(v2)

    bond_length, bond_angle = calculate_internal_coordinates(pose, residue_number)

    new_x = bond_length * np.cos(bond_angle)
    new_y = bond_length * np.sin(bond_angle)

    transformed_coord = atom_coords + new_x * v1 + new_y * v2

    print(f"Local transformed Coordinates for Residue {residue_number}:")
    for coord in transformed_coord:
        print(coord)
    
    return transformed_coord

# Example of transforming coordinates of each rotamer
for residue_number in range(1, pose.total_residue() + 1):
    local_coords = transform_to_local_coordinates(pose, residue_number)
   
for residue_number in range(1, pose.total_residue() + 1):
    local_coords = transform_to_internal_coordinates(coords, residue_number)


